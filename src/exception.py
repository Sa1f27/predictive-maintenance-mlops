import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        
        try:
            _, _, exc_tb = error_detail.exc_info()
            if exc_tb:
                file_name = exc_tb.tb_frame.f_code.co_filename
                line_number = exc_tb.tb_lineno
                self.error_message = f"Error in [{file_name}] line [{line_number}]: {str(error_message)}"
            else:
                self.error_message = f"Error: {str(error_message)}"
        except:
            self.error_message = f"Error: {str(error_message)}"

    def __str__(self):
        return self.error_message