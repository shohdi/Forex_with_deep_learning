# import sqlite3 module
import sqlite3

# define a class named Database
class LiteDatabase:

    # define the __init__ () method as the constructor
    def __init__ (self, db_name, table_name):

        # assign the parameters to the instance attributes
        self.db_name = db_name
        self.table_name = table_name

        # create a connection object to the database file
        self.conn = sqlite3.connect(self.db_name)

        # create a cursor object to execute SQL commands
        self.cur = self.conn.cursor()

        # create a table with the columns ID, lastStepTime, buyPercentage, isOpen, ShouldSendMail, and IsSentMail
        self.cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            ID TEXT PRIMARY KEY,
            lastStepTime TEXT,
            buyPercentage REAL,
            isOpen INTEGER,
            ShouldSendMail INTEGER,
            IsSentMail INTEGER
        )
        """)

        # commit the changes to the database
        self.conn.commit()
        self.close_connection()

    # optionally, define other methods to perform other operations on the database
    # for example, a method to insert data into the table
    def insert_data (self, id, last_step_time, buy_percentage, is_open, should_send_mail, is_sent_mail):
        # create a connection object to the database file
        self.conn = sqlite3.connect(self.db_name)

        # create a cursor object to execute SQL commands
        self.cur = self.conn.cursor()
        # execute an INSERT statement with the values
        self.cur.execute(f"""
        INSERT INTO {self.table_name} (ID, lastStepTime, buyPercentage, isOpen, ShouldSendMail, IsSentMail)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (id, last_step_time, buy_percentage, is_open, should_send_mail, is_sent_mail))

        # commit the changes to the database
        self.conn.commit()
        self.close_connection()

    # optionally, define a method to close the connection
    def close_connection (self):

        # close the connection
        self.conn.close()



    # define a method to run a query string against the table and return a list of the data model class
    def query_data (self, query):
        # create a connection object to the database file
        self.conn = sqlite3.connect(self.db_name)

        # create a cursor object to execute SQL commands
        self.cur = self.conn.cursor()

        # execute the query string
        self.cur.execute(query)

        # fetch the results as a list of tuples
        data = self.cur.fetchall()

        self.close_connection()

        # return the data
        return data
    

    # define a method to update values by id using the same data model class
    def update_data_by_id (self, id, last_step_time, buy_percentage, is_open, should_send_mail, is_sent_mail):
        # create a connection object to the database file
        self.conn = sqlite3.connect(self.db_name)

        # create a cursor object to execute SQL commands
        self.cur = self.conn.cursor()

        # execute an UPDATE statement with the new values and the id
        self.cur.execute(f"""
        UPDATE {self.table_name} SET lastStepTime = ?, buyPercentage = ?, isOpen = ?, ShouldSendMail = ?, IsSentMail = ? WHERE ID = ?
        """, (last_step_time, buy_percentage, is_open, should_send_mail, is_sent_mail, id))

        # commit the changes to the database
        self.conn.commit()
        self.close_connection()



    # define a method to get values by id in a data model class
    def get_data_by_id (self, id):
        # create a connection object to the database file
        self.conn = sqlite3.connect(self.db_name)

        # create a cursor object to execute SQL commands
        self.cur = self.conn.cursor()
        # execute a SELECT statement with the id
        self.cur.execute(f"""
        SELECT * FROM {self.table_name} WHERE ID = ?
        """, (id,))

        # fetch the result as a tuple
        data = self.cur.fetchone()
        self.close_connection()
        # return the data
        return data


