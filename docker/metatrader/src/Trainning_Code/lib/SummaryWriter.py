import tensorflow as tf
import datetime
import os


class SummaryWriter:
    def __init__(self,comment):
        
        self.comment = comment
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = os.path.join( 'logs' , current_time + comment)
        self.writer = tf.summary.create_file_writer(self.train_log_dir)

    def close(self):
        return
    


    def add_scalar(self,name,data,step):
        
           
        with self.writer.as_default():
            tf.summary.scalar(name,data, step=step)
        
    
    



    