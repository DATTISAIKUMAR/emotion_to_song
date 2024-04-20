import sqlite3


conn=sqlite3.connect('login.db')
cursor=conn.cursor()
cursor.execute('''
              CREATE TABLE IF NOT EXISTS login(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT NOT NULL UNIQUE,
                  password TEXT NOT NULL)
              ''')

conn.commit()
conn.close()




def create_connection():
    conn = sqlite3.connect('users.db')
    return conn

# Function to create user table if not exists
def create_user_table(conn):
    sql_create_users_table = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL,
        password TEXT NOT NULL
    );
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql_create_users_table)
    except sqlite3.Error as e:
        print(e)

# Create database connection and table
conn = create_connection()
create_user_table(conn)
conn.close()
