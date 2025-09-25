from fastapi import FastAPI
from pydantic import BaseModel
import pymysql

# FastAPI app
app = FastAPI()

# MySQL connection function
def get_connection():
    return pymysql.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="Benchmatrix@1234",
        database="aiprojects"
    )

# Pydantic models
class User(BaseModel):
    username: str
    password: str

class UpdateUser(BaseModel):
    id: int
    new_username: str
    new_password: str

class DeleteUser(BaseModel):
    id: int

@app.post("/add_user")
def add_user(user: User):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (user.username, user.password)
        )
        conn.commit()

        return {"status": "success", "message": f"User {user.username} added!"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if 'conn' in locals() and conn.open:
            conn.close()

@app.get("/users")
def get_users():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id, username, password FROM users")
        rows = cursor.fetchall()

        users = [{"id": row[0], "username": row[1], "password": row[2]} for row in rows]

        return {"users": users}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if 'conn' in locals() and conn.open:
            conn.close()

@app.put("/update_user")
def update_user(data: UpdateUser):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE users SET username = %s, password = %s WHERE id = %s",
            (data.new_username, data.new_password, data.id)
        )
        conn.commit()

        if cursor.rowcount == 0:
            return {"status": "error", "message": f"No user found with id {data.id}"}

        return {"status": "success", "message": f"User {data.id} updated!"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if 'conn' in locals() and conn.open:
            conn.close()

# 🔹 DELETE method
@app.delete("/delete_user")
def delete_user(data: DeleteUser):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM users WHERE id = %s", (data.id,))
        conn.commit()

        if cursor.rowcount == 0:
            return {"status": "error", "message": f"No user found with id {data.id}"}

        return {"status": "success", "message": f"User {data.id} deleted!"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if 'conn' in locals() and conn.open:
            conn.close()
