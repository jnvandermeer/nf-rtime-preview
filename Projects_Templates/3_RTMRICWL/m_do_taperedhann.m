%
% 'overseer' function for performing & logging subtraction in a tapered
% approach.
%
% [data,cfg]=m_do_taperedhann(data,cfg)

function [data,cfg]=m_do_taperedhann(data,cfg)


% this code needs some severe cleanup.
% several variables are unused, and some matrices could be defined as a bit
% smaller.


verbose=0;

cwregression=cfg.cwregression;

% as input we have the data matrix, and the regressor matrix.
% we have sampling rate
% we have the length of the window in sec
% we have the amount of temporal delay in sec

function_to_calculate_nwindows=@(x) 2*(2^x-1)+1;
nwindows=function_to_calculate_nwindows(cwregression.taperingfactor);
% do some complicated arithmatics to make sure that the boundaries of the
% tapering windows always falls on a sample (and not between samples, as
% then the sum would no longer hold...
% iteratively add (length) in samples to obtain this result.
taper_factor = cwregression.taperingfactor;

number_of_samples_in_window = floor(cwregression.srate*cwregression.windowduration + 1);
window=cwregression.taperingfunction(number_of_samples_in_window);
while rem((number_of_samples_in_window-1)/2^taper_factor,1)>0
    number_of_samples_in_window = number_of_samples_in_window + 1;
end
nsteps = 2^taper_factor;
step_in_samples = (number_of_samples_in_window - 1) / nsteps;
% division_factor = taper_factor;

% prepare where the window(s) should begin...
begins_segments = [1];
for i=1:(nsteps-1)
    begins_segments(end+1) = begins_segments(end) + step_in_samples;
end

delay_in_samples = floor(cwregression.srate*cwregression.delay);
division_factor = 2^(taper_factor-1);

x=data.matrix(cwregression.channelinds,:);
regs=data.matrix(cfg.cwregression.regressorinds,:);

if taper_factor==0
    disp('no subtraction will occur; taper factor is zero (should be >=1)!');
end


% reserve memory... for the fitted regs, and their weights...
matrix_stored_fits = zeros(numel(cwregression.channelinds),number_of_samples_in_window,nwindows,class(regs));
matrix_stored_weights = zeros(number_of_samples_in_window,nwindows,class(regs));


% sine-and-cosine;
% a taper factor of 0 = no window whatsoever!
% can be also 3, or 4 even.

% decide how many window need to be stored.
% --> nwindows

% determine division factor.
% --> division_factor

% determine the step size... and if that can be done, given the length of
% the window.
% --> step_in_samples

% determine what the points are where a new window should begin/need to be
% corrected or accounted for.
% --> begins_correction = 1:step_in_samples:size(data,2)


% if there's still enough data...
% for now... just store the subtracted data, so I can view it...
subtracted_signals = zeros(size(x),class(x));

subtracted_signals_weights = zeros(size(x,2),1,class(x));

% stores logging (fitparameters, etc).
store_logging={};

% this will initialize the piece of memory for storing where you sum over
% the windows...
summation=zeros(numel(cwregression.channelinds),step_in_samples+1,class(x));

% for me to check.. if things add up nicely!!
% for me to check... weights
summation_weights=zeros(step_in_samples+1,1,class(x));

% this was for plotting it!!
summatrix_check=zeros(2,step_in_samples+1,class(x));


% if you later decide to skip certain bad fits/windows, the division needs
% to be accounted for separately. Since this is a bit more complicated, we
% now divide (see later on) by 2^(taper_factor-1) and leave it at that.
% for this purpose, matrix_weights_fits would exist (see commented code).
jcheck=0;
max_windows = round(size(data.matrix,2) / step_in_samples);
current_sample = 1;
try
while current_sample < size(x,2) - number_of_samples_in_window;
    
    jcheck=jcheck+1;
    fprintf('doing window: %d out of %d \n',jcheck,max_windows);
    
    % what does this select??
    range = current_sample:(current_sample+number_of_samples_in_window-1);
    
    
    % decide if I can already do subtraction; if so..
    if current_sample>=number_of_samples_in_window
        
        % keyboard;
        % determine the signal I must subtract; this is summation.
        summation(:)=0;
        
        % for me to check... weights
        summation_weights(:)=0;
        for i=1:2^taper_factor
            % sumrange=(number_of_samples_in_window-(i*step_in_samples)) : number_of_samples_in_window-(i-1)*step_in_samples;
            
            % from the previous window, take the first part,
            % from the window before that, take the second part
            % from the window before that (even), takke the third part
            % and so on, etc, etc... until you have summed it all nicely.
            % keyboard;
            
            % from first window, take end samples, from second window, take
            % end-1 part of samples, etc, etc.
            % from the first STORED window (which is the last window
            % actually), take the FIRST part!!
            % sumrange = (number_of_samples_in_window - i*step_in_samples):((number_of_samples_in_window - (i-1)*step_in_samples));
            sumrange=(((i-1)*step_in_samples):(i*step_in_samples))+1;
            
            summation=summation + matrix_stored_fits(:,sumrange,i);
            % for me to check... weights
            summation_weights=summation_weights + matrix_stored_weights(sumrange,i);
            
            % this is just for me to check..
            % this was it... the windows have not (yet) been shifted; so,
            % it was correct what I was doing, only I took the wrong
            % windows.
            % let's see how things turn out now.
            % and of course, in the beginning I take only the rising flank
            % and the rest was 0.
            summatrix_check(i,:) = matrix_stored_fits(1,sumrange,i);
            
            % summatrix_check(i,:) = matrix_stored_weights(sumrange,i);
            
            
        end
        % again, just for me to check it...
        if verbose
        if jcheck<8
            figure;plot(summatrix_check');
            ylim([-0.5 0.5]);
            legend({'1','2','3','4'});
            jcheck=jcheck+1;
        end
        end
        
        % divide it by how much the sum should be (!)
        summation=summation/2^(taper_factor-1);
        summation_weights=summation_weights/2^(taper_factor-1);
        
        % subtract that signal from the data.. in the correct range!!
        subtractrange = (current_sample-step_in_samples+1):current_sample;
        
        % yes, it's 2 and not 1; because otherwise I would doubly correct
        % things. should not matter, though, due to the tapering approach.
        subtracted_signals(:,subtractrange) = summation(:,2:end);
        
        % for me to check weights...
        subtracted_signals_weights(subtractrange) = summation_weights(2:end);
        
    end
    
    
    % do a new window;
    xpart = x(:,range);
    regspart = regs(:,range);
    [fittedregs logging]=tools.fit_regmat_to_signalmat(xpart,regspart,window,delay_in_samples,[]);
    % do fifo rule! (% shift the windows backwards)
    % shift it (carefully, without (hopefully) overwriting stuff.
    for im=(size(matrix_stored_fits,3)):-1:2;
        matrix_stored_fits(:,:,im)=matrix_stored_fits(:,:,im-1);
        % for putting in the weights... and for me to check!!
        matrix_stored_weights(:,im)=matrix_stored_weights(:,im-1);
    end
    % keyboard;
    matrix_stored_fits(:,:,1) = fittedregs;
    % put hann windows in there...
    matrix_stored_weights(:,1) = window;
    % matrix_weights_fits(:,1:end-1) = matrix_weights_fits(:,2:end);
    % matrix_weights_fits = window;
    
    % store fitting parameters
    store_logging{end+1} = logging;
    
    
    % annotate begin/end
    % annotate all that fitted data...
    
    % I can check which part is complete...
    
    % increment our little while loop counter.
    current_sample = current_sample+step_in_samples;
    
end
catch
    keyboard;
end


cfg.cwregression.logging = store_logging;
data.subtracted_data = subtracted_signals;