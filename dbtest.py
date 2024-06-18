import sqlite3

conn = sqlite3.connect('test.db')

cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS DetectedItems (
        Person TEXT,
        Type TEXT,
        LicensePlate TEXT
    )""")


detected_items = [
    ['ALI:0.85', 'rectangle license plate', 'KL23AH5674'],
    ['ALI:0.92', 'rectangle license plate', 'MH12AB1234'],
    # ... more detected items
]

for item in detected_items:
    cur.execute('''
        INSERT INTO DetectedItems (Person, Type, LicensePlate)
        VALUES (?, ?, ?)
    ''', item)


for row in cur.execute('SELECT * FROM DetectedItems'):
    print(row)


conn.commit()
conn.close()