import sqlite3

print("user table")
conn=sqlite3.connect('users.db')
cursor = conn.cursor()
cursor.execute('Select * from users')
rows=cursor.fetchall()
if not rows:
	print("No users existed")
for i in rows:
	print(i)
conn.commit()
conn.close()


print("table login")

conn=sqlite3.connect('login.db')
cursor = conn.cursor()
cursor.execute('Select * from login')
rows=cursor.fetchall()
if not rows:
	print("No users existed")
for i in rows:
	print(i)
conn.commit()
conn.close()