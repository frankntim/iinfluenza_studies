import sqlite3
import random
import datetime
def create_customer_database(db_name="customer_profiles.db"):
    """
    Creates an SQLite database with Customer, Contact, Customer Address, and Address tables.
    Modified Customer table to include Source_System and Gender.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create Customer Table (Modified)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Customer (
                CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
                FirstName TEXT NOT NULL,
                LastName TEXT NOT NULL,
                DateOfBirth TEXT,
                Email TEXT UNIQUE,
                Source_System TEXT,
                Gender TEXT
            );
        """)

        # Create Contact Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Contact (
                ContactID INTEGER PRIMARY KEY AUTOINCREMENT,
                CustomerID INTEGER,
                PhoneNumber TEXT,
                ContactType TEXT, -- e.g., Mobile, Home, Work
                FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
            );
        """)

        # Create Address Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Address (
                AddressID INTEGER PRIMARY KEY AUTOINCREMENT,
                StreetAddress TEXT,
                City TEXT,
                State TEXT,
                PostalCode TEXT,
                Country TEXT
            );
        """)

        # Create Customer Address Table (Linking table)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS CustomerAddress (
                CustomerAddressID INTEGER PRIMARY KEY AUTOINCREMENT,
                CustomerID INTEGER,
                AddressID INTEGER,
                FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID),
                FOREIGN KEY (AddressID) REFERENCES Address(AddressID)
            );
        """)

        conn.commit()
        print(f"Database '{db_name}' and tables created successfully.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

    finally:
        if conn:
            conn.close()

def insert_customer_data(db_name="customer_profiles.db"):
    """
    Inserts 20 sample customer records into the database.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        genders = ["Male", "Female", "Other"]
        sources = ["Web", "CRM", "Mobile App", "Social Media"]
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
        states = ["NY", "CA", "IL", "TX", "AZ", "PA"]

        for i in range(1, 21):  # Insert 20 records
            first_name = f"Customer{i}"
            last_name = f"Last{i}"
            dob = (datetime.date(random.randint(1950, 2000), random.randint(1, 12), random.randint(1, 28))).strftime("%Y-%m-%d")
            email = f"customer{i}@example.com"
            source = random.choice(sources)
            gender = random.choice(genders)

            cursor.execute("INSERT INTO Customer (FirstName, LastName, DateOfBirth, Email, Source_System, Gender) VALUES (?, ?, ?, ?, ?, ?)",
                           (first_name, last_name, dob, email, source, gender))

            street_address = f"{random.randint(100, 999)} Random St"
            city = random.choice(cities)
            state = random.choice(states)
            postal_code = f"{random.randint(10000, 99999)}"
            country = "USA"

            cursor.execute("INSERT INTO Address (StreetAddress, City, State, PostalCode, Country) VALUES (?, ?, ?, ?, ?)",
                           (street_address, city, state, postal_code, country))

            phone_number = f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
            contact_type = random.choice(["Mobile", "Home", "Work"])

            cursor.execute("INSERT INTO Contact (CustomerID, PhoneNumber, ContactType) VALUES (?, ?, ?)",
                           (i, phone_number, contact_type))

            cursor.execute("INSERT INTO CustomerAddress (CustomerID, AddressID) VALUES (?, ?)", (i, i)) #assuming 1 to 1 customer address relationship for simplicity.

        conn.commit()
        print("20 Sample customer records inserted successfully.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

    finally:
        if conn:
            conn.close()

def query_customer_data(db_name="customer_profiles.db"):
    """
    Queries and prints customer data joined with address and contact details.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT Customer.FirstName, Customer.LastName, Address.StreetAddress, Address.City, Contact.PhoneNumber, Customer.Source_System, Customer.Gender
            FROM Customer
            JOIN CustomerAddress ON Customer.CustomerID = CustomerAddress.CustomerID
            JOIN Address ON CustomerAddress.AddressID = Address.AddressID
            JOIN Contact ON Customer.CustomerID = Contact.CustomerID;
        """)

        rows = cursor.fetchall()
        if rows:
            print("\nCustomer Data:")
            for row in rows:
                print(f"Name: {row[0]} {row[1]}, Address: {row[2]}, {row[3]}, Phone: {row[4]}, Source: {row[5]}, Gender: {row[6]}")
        else:
            print("No customer data found.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_customer_database()
    insert_customer_data()
    query_customer_data()