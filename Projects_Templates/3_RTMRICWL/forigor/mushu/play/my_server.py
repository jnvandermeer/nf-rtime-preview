
# we need asyncio, apparently?
import asyncio 

from multiprocessing import Process, Queue, Event
import logging
import time


def handle_data(queue,data):
    # do the time-stamp thingy + put it into the queue.
    timestamp=time.time()
    markertext  = data.decode("utf-8")
    queue.put([timestamp, markertext])
    print("%d" % queue.qsize())
    
    # then print it out... hehe
    # item=queue.get()
    # queue.put(item)
    # print(item)
    

def marker_reader(queue, running, ready):
	
    # define our UDP class , which includes what to actually DO with the data:
    # this also kind-of uses the async/await stuff!

    # so if I wanna pass stuff on to a queue, I'd have to give the queue as an input argument, so we can put stuff on it, right?
    # put the queue in dunder init
    # why isn't this subclassed?
    # copy/paste the following from the examples on asyncio in python docs. We also tested these.
    class EchoServerProtocol(asyncio.DatagramProtocol):

        # i'll just define a setter method to pass on the queue. Man.
        def __init__(self, queue):
            self.queue=queue
         
        def set_queue(self,queue):
            self.queue.queue        
        
        
        def connection_made(self, transport):
            self.transport = transport
    
        def datagram_received(self, data, addr):
            message = data.decode()
            print('Received %r from %s' % (message, addr))
            print('Send %r to %s' % (message, addr))
            self.transport.sendto(data, addr)
            
            # so -- instead of echo-ing some stuff -- move the queue forward to
            # -- or IN ADDITION TO echo-ing --> handle the data.
            # data handler, along with the data.
            handle_data(queue, data)
    
    # Define also the TCP class, with also a 'data handler'
    # put the queue in dunder init
    class EchoServerClientProtocol(asyncio.Protocol):
        
        def __init__(self, queue):
            self.queue=queue
        #  # do I need to do this?
        #asyncio.Protocol.__init__(self)
        
        
        def set_queue(self,queue):
            self.queue=queue
        
        def connection_made(self, transport):
            peername = transport.get_extra_info('peername')
            print('Connection from {}'.format(peername))
            self.transport = transport
    
        def data_received(self, data):
            message = data.decode()
            print('Data received: {!r}'.format(message))
    
            print('Send: {!r}'.format(message))
            self.transport.write(data)
    
            print('Close the client socket')
            self.transport.close()
            
            # OK and then:
            handle_data(queue, data)
            
    
    
    # get our event loop from asyncio -- not our own event loop stuff as shown in the youtube video:
    # it's still a bit esotheric.
    loop = asyncio.get_event_loop()
    
    
    # add stuff to the loop. First the UDP:
    # One protocol instance will be created to serve all client requests
    # transport is a coroutine?
    print("Starting UDP server")
    listen = loop.create_datagram_endpoint(
        lambda: EchoServerProtocol(queue), local_addr=('127.0.0.1', 6666))
    transport, protocol = loop.run_until_complete(listen)
    
    # then the TCP: -- why one is called a coro, and another is called 'listen', I do not know. 
    # I also wish to check whether 
    print("Starting TCP server")
    # the docs of asyncio are abhorrent. According to docs, this returns a server object.
    # but a server object is apparently also a coroutine?
    coro = loop.create_server(lambda: EchoServerClientProtocol(queue), '127.0.0.1', 6666)
    server = loop.run_until_complete(coro)
    
    # hmmm... we CAN change things so that TCP and UDP should use different ports.
    # do UDP and TCP use different ports to begin with??
    # OK... so this works quite well. Was easier than I thought. It all works nicely concurrently, too.
    # so I just need to Methodify this, match to what's written in mushu, and pass on the multiprocessing queue?
    
    # try that...
    
    # say that we're ready.
    ready.set()
    
    
    # future=Future()
    
    print("starting...")
    try:
        print('test')
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    print("stopping...")
    
    
    
    
    # closing statements -- do we need those?:
    # close the transport:
    # transport.close() .. hmm apparently, we won't have to 'close' the transport?
    # loop.run_until_complete(transport.wait_closed()) # does this make sense --> NO
    
    # Close the server
    # TCP:
    
    
    server.close()
    loop.run_until_complete(server.wait_closed())
    
    
    # close the loop, plz?
    loop.close()



if __name__ == "__main__":
	
    # start the logger:	
    logger = logging.getLogger(__name__)
    logger.info('Logger started')
    
    
    # do our little queue:
    queue = Queue()
    
    # what to do with f.e. running and ready?
    # running is the first multiprocessing.event, and it is SET, so...
    running = asyncio.Event()
    running.set() 
    
    # the second event is just an event and it's set later on in the declaration(s):
    ready = Event()
    
    # call our function:
    # marker_reader(queue, running, ready)
    
    P = Process(target=marker_reader, args=(queue, running, ready))
    P.start()
    
    
    # time.sleep(1)
    
    # print('tja')
    
    # try to stop the main event loop?
    running.clear()
    P.join()
    
    