import socket
import sys
import code

class StringOutput:
    def __init__(self):
        self.output = ''

    def write(self, s):
        self.output += s

    def flush(self):
        pass
    
local_vars = {}

def handle_connection(client_socket):
    
    
    string_output = StringOutput()
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = string_output
    sys.stderr = string_output
        
    data = client_socket.recv(100000).decode()
    
    # e.g: [p,r,i,n,t,(,',h,i,',)] -> print('hi')
    data = data[1::2]
    
    '''
    mode = [FILE] -> run code in FILE
    mode = [COMMAND] -> run code in python interpreter
    '''
    
    if data.startswith("[FILE]"):
        file_path = './' + data[6:]
        mode = "[FILE]"
    else:
        mode = "[COMMAND]"
        console = code.InteractiveConsole(locals=local_vars)
    try:
        if mode == "[FILE]":
            with open(file_path, 'r') as file:
                Code = file.read()
                exec(Code)
        elif mode == "[COMMAND]":
            console.push(data)
        
        # remove "\n"
        result = string_output.output[:-1]
        
    except Exception as e:
        result = str(e)
        
    finally:
       
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    client_socket.sendall(result.encode())

    client_socket.close()
    
    
def run_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('127.0.0.1', 8888)
    server_socket.bind(server_address)
    server_socket.listen(1)
    print(f"Python server listening on {server_address}")

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connected to {client_address}")
        handle_connection(client_socket)

if __name__ == "__main__":
    run_server()
