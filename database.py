import sqlite3

DATABASE_NAME = "crop_predictions.db"


def create_database():
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER NOT NULL,
            state TEXT NOT NULL,
            crop TEXT NOT NULL,
            season TEXT NOT NULL,
            area REAL NOT NULL,
            rainfall REAL NOT NULL,
            fertilizer REAL NOT NULL,
            pesticide REAL NOT NULL,
            predicted_yield REAL NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(
    year,
    state,
    crop,
    season,
    area,
    rainfall,
    fertilizer,
    pesticide,
    predicted_yield
):
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            year,
            state,
            crop,
            season,
            area,
            rainfall,
            fertilizer,
            pesticide,
            predicted_yield
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        year,
        state,
        crop,
        season,
        area,
        rainfall,
        fertilizer,
        pesticide,
        predicted_yield
    ))

    connection.commit()
    connection.close()


def get_predictions():
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """)

    predictions = cursor.fetchall()

    connection.close()

    return predictions

def get_prediction_summary():
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            COUNT(*),
            AVG(predicted_yield)
        FROM predictions
    """)

    total_predictions, average_yield = cursor.fetchone()

    cursor.execute("""
        SELECT
            year,
            state,
            crop,
            season,
            predicted_yield
        FROM predictions
        ORDER BY id DESC
        LIMIT 1
    """)

    latest_prediction = cursor.fetchone()

    connection.close()

    return {
        "total_predictions": total_predictions,
        "average_yield": average_yield,
        "latest_prediction": latest_prediction
    }

def get_crop_summary():
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            crop,
            COUNT(*) AS prediction_count,
            AVG(predicted_yield) AS average_yield
        FROM predictions
        GROUP BY crop
        ORDER BY prediction_count DESC
    """)

    crop_summary = cursor.fetchall()

    connection.close()

    return crop_summary

def delete_prediction(prediction_id):
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM predictions
        WHERE id = ?
    """, (prediction_id,))

    connection.commit()
    connection.close()

def search_predictions(search_text="", season=""):
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    query = """
        SELECT *
        FROM predictions
        WHERE
            (
                state LIKE ?
                OR crop LIKE ?
            )
    """

    search_pattern = f"%{search_text}%"

    parameters = [
        search_pattern,
        search_pattern
    ]

    if season:
        query += " AND season = ?"
        parameters.append(season)

    query += " ORDER BY id DESC"

    cursor.execute(query, parameters)

    predictions = cursor.fetchall()

    connection.close()

    return predictions