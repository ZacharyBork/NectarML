import argparse
from nectarml.viz import web    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Starts a web visualization server.')
    parser.add_argument(
        '--host', type=str, default='localhost', 
        help='The host address for the server.')
    parser.add_argument(
        '-p', '--port', type=int, default=8097, 
        help='The port number to open for the server.')
    args = parser.parse_args()

    server = web.Server(host=args.host, port=args.port)
    server.run()
    
