"""Create (or reset) a login account.  Usage:  python create_user.py"""
import getpass

import auth

username = input("Username: ").strip()
password = getpass.getpass("Password (at least 6 characters): ")
again = getpass.getpass("Repeat password: ")
if password != again:
    raise SystemExit("Passwords do not match. Nothing was saved.")
role = input("Role [staff/admin] (press Enter for staff): ").strip().lower() or "staff"
auth.add_user(username, password, role)
print(f"Account '{username.lower()}' saved. The password is stored hashed, never in plain text.")
