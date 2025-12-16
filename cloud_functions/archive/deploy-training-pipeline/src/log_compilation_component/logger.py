'''
 # @ Create date: 2024-02
 # @ Modified date: 2024-02-09
 '''

class Logger(object):
    '''
    This is a simple logging class that redirects standard output to both the terminal
    and a log file. This ensures logs from the pipeline job are viewable both 
    on Vertex AI's console and easily exportable.
    '''
        
    def __init__(self, file_name, original_stdout):
        self.terminal = original_stdout
        self.log = open(file_name, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        try:
            self.log.flush()  
        except Exception as e: 
            print(f'Error in write()')
            print(f'Message: {message}')

    def flush(self):
        try:
           self.log.flush()
        except Exception as e: 
            print(f'Error in flush()')

    def close(self):
        self.log.close()

