from libmushu.driver.bptools.bptools import Receiver
from libmushu.driver.bptools.container import DataContainer

import multiprocessing
import asyncio
import time


# f1 --> grab stuff from the receiver queue, put it into the container.
# the container class functions as the buffer. it's basically a numpy array.
async def put_into_container(container, queue_incoming):
    # make the container handle the queue item. In other words, the Receiver
    # captured some data and put it into the queue. This will use this process's
    # computational resources to make the container handle that new data
    # package.
    while True:

        while not queue_incoming.empty():


            queue_item = queue_incoming.get()

            # print(queue_item)

            if 'hdr' in queue_item.keys():
                container.pass_hdr(queue_item['hdr'])
            if 'd' in queue_item.keys():
                # print queue_item['block']
                container.pass_data(queue_item['d'], queue_item['markers'], queue_item['block'])

        # and.. yield back the control.
        await asyncio.sleep(0.00001)


# f2 --> check if there's a 'get_data' instruction in the queue, and then deal with that.
# this function will actually become bigger in the future to allow for different commands, too.
async def get_from_container(container, queue_instructions, queue_data, datasent):
    # check whether there's anything in the queue.

    printed_header_warning = False


    while True:


        while queue_instructions.qsize() > 0:

            while not container.pass_data_initialized or not container.pass_hdr_initialized:
                if not printed_header_warning:
                    print('We will wait until the header and a small part of data has been passed on to')
                    print('container by datacurator. If this takes too long, you should debug.')
                    print('See the flowchart')
                    printed_header_warning = True
                await asyncio.sleep(0.00001)

            #print('returning some data')
            #print(container.pass_data_initialized)
            #print(container.pass_hdr_initialized)

            # handle queue requests
            queue_item = queue_instructions.get()
            # print(queue_item[0])

            if queue_item[0] == 'get_data':

                if queue_item[1] is not -1:
                    interval = queue_item[1]

                    # figure out the current start/stop, but do not (yet) change anything
                    # regarding current positions, so as to allow get_data to still get unobtained
                    # items...

                    # print('hallo!')
                    # print(interval[0])
                    # print(interval[1])

                    data = container[interval[0]: interval[1], :]

                    markers = container.obtain_markers([interval[0], interval[1]])
                    #print(interval[0])
                    #print(interval[0])

                    if data is not None:
                        queue_data.put((data, markers))
                    datasent.set()

                else:

                    # probably can do better --> implement this in container?
                    points_in_block = container.get_samples_in_block()

                    # print('OK... so... now we shall obtain some data.')

                    # this is the interaction --> a counter in container which gets updated
                    nblocks = container.get_last_block() - container.get_block_position()

                    # print(nblocks)
                    # print(nblocks * points_in_block)

                    if nblocks > container.max_request_blocksize:
                        nblocks = container.max_request_blocksize
                        print('i can give only %d blocks, since it was too long since get_data was called' % nblocks)
                        # nblocks = 5000


                    if nblocks > 0:
                        data = container[0: (nblocks * points_in_block), :]

                        markers = container.obtain_markers([0, (nblocks * points_in_block)])

                        queue_data.put((data, markers))

                        container.set_block_position(container.get_last_block())

                        # tell the other (i.e. main) process that this function has done its job.
                        datasent.set()
                    else:
                        # abd here is where it Jams, since later on (wit tries to unpack - it's nto working anymore.

                        data = container[0:0, :]  # i.e., NO data.
                        markers=[]
                        queue_data.put((data, markers))  # stop it from jamming?
                        datasent.set()


            if queue_item == 'get_hdr':
                queue_data.put(container.hdr)
                datasent.set()

        # yield back control...
        await asyncio.sleep(0.00001)

    # yield back control...
    await asyncio.sleep(0.00001)


async def check_stop_loop(loop, killswitch):
    while not killswitch.is_set():
        await asyncio.sleep(0.00001)
    loop.stop()


class DataCurator(multiprocessing.Process):
    """
    So this class will be the Process in between the Main Process and the Receiver Process
    It will use Queues for communication
    R --> DC --> Main
             <-- Main
    DC contains the Container, which contains the NP Array.
    This is a way of having the continuous Data Buffer ready
    AND also put all of this upkeep and shit in separate Processes
    Freeing up my Main Process to do all of the Fancy Stuff without
    having to worry about data buffering or keeping track of things.

    So this is a framework that should also work with other Amp Receivers

    This should start an asyncio event loop, and in order to nicely shut down
    this main event loop -- it should accept a

    """

    def __init__(self, ip_address, port, backgroundbuffsize):
        super(DataCurator, self).__init__()

        self.ip_address = ip_address
        self.port = port
        self.backgroundbuffsize = backgroundbuffsize


        self.queue_instructions = multiprocessing.Queue()
        self.queue_data = multiprocessing.Queue()

        # and a toggle to help killing our Process.
        self.killswitch = multiprocessing.Event()
        self.datasent = multiprocessing.Event()
        # self.header_is_handled = False

    def run(self):

	# All code in here runs in the subprocess, with a memory footprint of the rest of the stuff in this class.
        print('OK .. starting the DataCurator.')
        queue_incoming = multiprocessing.Queue()   # this is container's incoming, but receiver's outgoing

        receiver = Receiver(queue_incoming, self.ip_address, self.port)
        # it'd actually be 'better' to more explicitly use variables here, instead of relying on the self. variables.
        receiver.start()
        # time.sleep(1)

        # set things up... the DataContainer LIVES inside the subprocess and we use the queues to communicate with them.
        container = DataContainer(maxlength_in_seconds=self.backgroundbuffsize)
        # a toggle for helping out whether we've sent data

        loop = asyncio.new_event_loop()
        loop.create_task(put_into_container(container, queue_incoming))
        loop.create_task(get_from_container(container, self.queue_instructions, self.queue_data, self.datasent))
        loop.create_task(check_stop_loop(loop, self.killswitch))
        # loop.create_task(self.get_data())
        print(__name__)
        print('Loop Starting.')
        loop.run_forever()

	# since we've arrived here -- we've escaped from the main event loop.
	# this is in essence a while loop (even though you don't see it here.
        print('loop stopped!')

        # once you're here -- I guess the loop as stopped, since normally python jams around here.
        # so once the loop has stopped -- also stop the Receiver (from clogging the incoming queue.
        receiver.shutdown()
        print('receiver shutdown sent!')
        receiver.join()
        print('receiver successfully joined!')

    def stop_acquisition(self):
        print('requesting to stop the acquisition')
        self.killswitch.set()

    @property
    def is_data_sent(self):
        if self.datasent.is_set():
            return True
        return False

    def get_data(self):

        # so! since this lives still inside the Master Process (and the Slave Process is the stuff/mem c
        # copy of whats defined in this class --> it'd be trivial to get some data, right?

        # so IF we wish to return some data --> just put something into a queue, set returndata false
        # and whenever returndata == true --> return the data.
        # hopefully the other process will be fast enough so as to do it fastly.
        self.datasent.clear()
        self.queue_instructions.put(('get_data', -1))
        while not self.datasent.is_set():  # yeah ... you safeguard by putting only 1 item in queue and then setting
                                            # the datasent multiprocessing event.
                                            # JAM until datasent is set to True.
            time.sleep(0)  # effectively this is a 'pass' statement?

        # print(self.queue_data.qsize())

        #print('hallo!')
        #print(self.datasent.is_set())
        #print(self.queue_data.qsize())
        #print(self.queue_data.get())

        (data, markers) = self.queue_data.get()

        return data, markers

    def get_hdr(self):

        self.datasent.clear()
        self.queue_instructions.put('get_hdr')
        while not self.datasent.is_set():  # yeah ... you safeguard by putting only 1 item in queue and then setting
                                            # the datasent multiprocessing event.
                                            # JAM until datasent is set to True.
            time.sleep(0)  # effectively, a pass.

        # print(self.queue_data.qsize())
        return self.queue_data.get()

	# OR -- if you've done it already -- use the saved one.

# def starting_up_the_loop(container, queue_incoming, queue_instructions, queue_outgoing, curatorstop, datasent):
#    loop = asyncio.get_event_loop()
#    loop.create_task(put_into_container(container, queue_incoming))
#    loop.create_task(get_from_container(container, queue_instructions, queue_outgoing, datasent))
#    loop.create_task(kill_loop(loop, curatorstop))
#    loop.create_task(print_something())
#    loop.run_forever()


