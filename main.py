import time
import mediapipe as mp
import os
import pyttsx3
import threading
import shutil
from http import client
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field, EmailStr
import sqlite3
import json
import requests
import cv2
import numpy as np
from typing import Optional
import uvicorn

# Initialize a FastAPI application instance.
app = FastAPI()


# Load environment variables from a .env file and create an OpenAI client.
load_dotenv()
client = OpenAI(organization=os.environ.get("OPENAI_ORGANIZATION"), api_key=os.environ.get("OPENAI_API_KEY"))


def get_db_connection():
    """
           Establishes and returns a connection to the SQLite database.

           This function creates a connection to the SQLite database named 'Users.db'.

           Returns:
               sqlite3.Connection: A connection object to the SQLite database.
           """
    conn = sqlite3.connect("Users.db")
    conn.row_factory = sqlite3.Row
    return conn


def create_users_table():
    """
        Creates a 'Users' table in the SQLite database if it doesn't already exist.

        The table is designed to store information about users, including their
        email, height, weight, name, BMI, and their exercise records
        for deadlifts, squats, and bench presses. The 'email' field is used as the
        primary key for the table, ensuring that each email is unique across records.

        The function establishes a connection to the database using the `get_db_connection`
        helper function, executes the SQL command to create the table with the specified
        schema if it doesn't exist, commits the transaction, and then closes the connection.

        No parameters are needed for this function, and it does not return any value.
            """
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS Users (
        email TEXT PRIMARY KEY,
        height REAL NOT NULL,
        weight REAL NOT NULL, 
        name TEXT NOT NULL,
        bmi REAL NOT NULL,
        deadlift TEXT NOT NULL,
        squat TEXT NOT NULL,
        bench_press TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()


# Create the 'Users' table in the database if it doesn't already exist.
create_users_table()


class User(BaseModel):
    """
        A User model that defines the schema for user information.

        Attributes:
            email (EmailStr): The user's email address. It must be a valid email format.
            height (float): The user's height in meters. Must be greater than 0.
            weight (float): The user's weight in kilograms. Must be greater than 0.
            name (str): The user's full name.
        """
    email: EmailStr
    height: float = Field(gt=0, description="Height in meters")
    weight: float = Field(gt=0, description="Weight in kilograms")
    name: str


def calculate_bmi(height: float, weight: float) -> float:
    """
        Calculate and return the BMI.

        The BMI is calculated by dividing the weight in kilograms by the square of the height in meters.
        The result is rounded to two decimal places.

        Parameters:
        - height (float): The height of the individual in meters.
        - weight (float): The weight of the individual in kilograms.

        Returns:
        - float: The calculated BMI, rounded to two decimal places.
        """
    return round(weight / (height ** 2), 2)


@app.post("/Registration")
async def register_user(user: User):
    """
       Registers a new user in the database.

       This endpoint validates the incoming user data against several criteria (e.g., email format,
       positive height and weight, valid name) before inserting the data into the 'Users' table in the database.
       If the validation fails or if the email is already registered, it raises an HTTPException.

       Parameters:
       - user (User): A User model instance containing the user's email, height, weight, and name.

       Raises:
       - HTTPException: 400 error if any of the validation checks fail or if the email is already registered.

       Returns:
       - dict: A message confirming registration and the registered user data.
       """
    # Check for default or invalid values
    if user.email == "user@example.com":
        raise HTTPException(status_code=400, detail="Invalid email address provided.")

    if user.height <= 0.5:
        raise HTTPException(status_code=400, detail="Invalid height provided.")

    if user.weight <= 10:
        raise HTTPException(status_code=400, detail="Invalid weight provided.")

    if not user.name.strip() or user.name == "string" or user.name[0].isspace():
        raise HTTPException(status_code=400, detail="Invalid name provided.")

    conn = get_db_connection()

    # Prepare the data for insertion
    user_data = {
        "email": user.email,
        "height": user.height,
        "weight": user.weight,
        "name": user.name,
        "bmi": calculate_bmi(user.height, user.weight),
        # Initialize fields with default values
        "deadlift": json.dumps([]),
        "squat": json.dumps([]),
        "bench_press": json.dumps([])
    }
    try:
        conn.execute('''
            INSERT INTO Users (email, height, weight, name, bmi, deadlift, squat, bench_press) 
            VALUES (:email, :height, :weight, :name, :bmi, :deadlift, :squat, :bench_press)
            ''', user_data)
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")
    conn.close()
    return {"message": "Welcome to our exercise program", "user": user_data}


@app.get("/Show User")
async def get_user(user_email: EmailStr):
    """
        Retrieve and return user information by email.

        This endpoint fetches a user from the 'Users' database table based on the provided email address.
        It returns the user's details, including their exercise records for deadlifts, squats, and bench presses

        Parameters:
        - user_email (EmailStr): The email address of the user to retrieve.

        Returns:
        - A dictionary containing the user's details if the user is found. This includes the user's email,
          height, weight, name, BMI, and their exercise records.
        - Raises an HTTPException with status code 404 (Not Found) if no user is found with the provided email address.

        Raises:
        - HTTPException: A status code of 404 with detail "User not found" if the user does not exist in the database.
        """
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM Users WHERE email = ?', (user_email,)).fetchone()
    conn.close()
    if user:
        return {**user, "deadlift": json.loads(user["deadlift"]), "squat": json.loads(user["squat"]), "bench_press": json.loads(user["bench_press"])}
    else:
        raise HTTPException(status_code=404, detail="User not found")


@app.post("/Update Record Weight")
async def update_weight(email: EmailStr, new_deadlift: float = None, new_squat: float = None,
                         new_bench_press: float = None):
    """
        Update the exercise records for a user identified by their email address.

        This endpoint allows updating the user's deadlift, squat, and bench press records. If the user does not exist,
        a 404 error is returned. If invalid weights are provided (e.g., negative numbers), a 400 error is returned.

        Parameters:
        - email (EmailStr): The email address of the user whose records are to be updated.
        - new_deadlift (float, optional): The new deadlift record to add. Default is None.
        - new_squat (float, optional): The new squat record to add. Default is None.
        - new_bench_press (float, optional): The new bench press record to add. Default is None.

        Returns:
        - A dictionary with a message indicating the update's success.

        Raises:
        - HTTPException: A 404 error if no user is found with the provided email address.
        - HTTPException: A 400 error if any of the new weights are negative.

        Note:
        The function appends the new record to the existing list of records for each exercise
        (if the new record is provided and valid) and updates the user's record in the database.
        """
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM Users WHERE email = ?', (email,)).fetchone()

    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    squats = json.loads(user["squat"])
    bench_presses = json.loads(user["bench_press"])
    deadlifts = json.loads(user["deadlift"])

    if new_squat is not None:
        if new_squat < 0:
            conn.close()
            raise HTTPException(status_code=400, detail="Invalid weight of squat provided.")
        else:
            squats.append(new_squat)
            updated_squat_json = json.dumps(squats)
    else:
        updated_squat_json = user["squat"]

    if new_bench_press is not None:
        if new_bench_press < 0:
            conn.close()
            raise HTTPException(status_code=400, detail="Invalid weight of bench press provided.")
        else:
            bench_presses.append(new_bench_press)
            updated_benchpress_json = json.dumps(bench_presses)
    else:
        updated_benchpress_json = user["bench_press"]

    if new_deadlift is not None:
        if new_deadlift < 0:
            conn.close()
            raise HTTPException(status_code=400, detail="Invalid weight of deadlift provided.")
        else:
            deadlifts.append(new_deadlift)
            updated_deadlift_json = json.dumps(deadlifts)
    else:
        updated_deadlift_json = user["deadlift"]

    # Update the user's weight, bmi, and weight loss percentage in the database
    conn.execute('UPDATE Users SET squat = ?, deadlift = ?, bench_press = ? WHERE email = ?',
                 (updated_squat_json, updated_deadlift_json, updated_benchpress_json, email))
    conn.commit()
    conn.close()
    return {"message": "User's records updated successfully"}


# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
# Initialize MediaPipe Drawing utilities
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    """
       Calculate the angle formed by three points.

       This function calculates the angle formed at point `b` by the line segments `a-b` and `b-c`. It uses the arctangent
       function (`arctan2`) to find the angle in radians between each pair of points and then converts this angle to degrees.
       If the calculated angle exceeds 180 degrees, it is adjusted to ensure the angle remains within the range [0, 180]
       degrees by subtracting it from 360 degrees.

       Parameters:
       - a (tuple[float, float]): The coordinates (x, y) of the first point.
       - b (tuple[float, float]): The coordinates (x, y) of the midpoint, where the angle is being measured.
       - c (tuple[float, float]): The coordinates (x, y) of the third point.

       Returns:
       - float: The angle in degrees formed at point `b` by the segments `a-b` and `b-c`.
       """
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def analyze_squat(landmarks):
    """
        Calculate the knee angle during a squat exercise using the positions of the hip, knee, and ankle landmarks.

        This function calculates the angle at the knee joint formed by the hip, knee, and ankle landmarks. These landmarks
        should correspond to the points on the left side of the body during a squat movement. The function utilizes the
        `calculate_angle` function to compute the angle in degrees, based on the coordinates of these three landmarks.

        Parameters:
        - landmarks (object): An object containing the detected pose landmarks. It should have attributes corresponding to
          the left hip, knee, and ankle, accessible through the `mp_pose.PoseLandmark` enumeration.

        Returns:
        - float: The calculated knee angle in degrees. This angle helps in analyzing the form and depth of the squat.
        """
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    knee_angle = calculate_angle(hip, knee, ankle)
    return knee_angle


def analyze_deadlift(landmarks):
    """
        Calculate the back angle during a deadlift exercise using the positions of the shoulder, hip, and knee landmarks.

        This function computes the angle formed at the hip joint by the line segments connecting the shoulder to the hip and
        the hip to the knee. These landmarks should correspond to points on the left side of the body during a deadlift
        movement. It leverages the `calculate_angle` function to determine the angle in degrees, based on the coordinates
        of these three landmarks.

        Parameters:
        - landmarks (object): An object containing the detected pose landmarks. It should have attributes corresponding to
          the left shoulder, hip, and knee, accessible through the `mp_pose.PoseLandmark` enumeration.

        Returns:
        - float: The calculated back angle in degrees. This angle is crucial for assessing the form of the deadlift,
          particularly in terms of maintaining a neutral spine to reduce the risk of injury.
        """
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    back_angle = calculate_angle(shoulder, hip, knee)
    return back_angle


def analyze_benchpress(landmarks):
    """
       Calculate the elbow angle during a bench press exercise using the positions of the wrist, elbow, and shoulder landmarks.

       This function evaluates the angle at the elbow joint formed by the wrist, elbow, and shoulder landmarks. These landmarks
       are intended to represent points on the left side of the body during a bench press movement. It utilizes the
       `calculate_angle` function to compute this angle in degrees, based on the coordinates of these three landmarks.

       Parameters:
       - landmarks (object): An object containing the detected pose landmarks. It should have attributes for
         the left wrist, elbow, and shoulder, accessible through the `mp_pose.PoseLandmark` enumeration.

       Returns:
       - float: The calculated elbow angle in degrees. This angle is essential for analyzing the form of the bench press,
         particularly in terms of elbow positioning relative to the body, which can affect the exercise's efficacy and safety.
       """
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow_angle = calculate_angle(wrist, elbow, shoulder)

    return elbow_angle


def analyze_exercise_form(video_path, exercise_type):
    """
        Analyzes the form of an exercise by calculating the angle of a specific joint or joints
        during the performance of squats, deadlifts, or bench presses from a video file.

        This function processes a video of an individual performing one of the specified exercises
        and calculates the relevant joint angles to assess form. For squats, it focuses on the knee angle;
        for deadlifts, the back angle; and for bench presses, the elbow angle. The function leverages
        MediaPipe's pose estimation to identify and track the required body landmarks across the video frames.

        Parameters:
        - video_path (str): The path to the video file to be analyzed. The video must be in .mp4 format.
        - exercise_type (str): The type of exercise being analyzed. Valid options are 'squat', 'deadlift', and 'benchpress'.

        Returns:
        - tuple: A tuple containing the final angle (float) calculated for the exercise and the exercise type (str).
          The angle returned is the minimum angle observed for squats and bench presses, and the average angle for deadlifts.

        Raises:
        - HTTPException: If the video file is not in .mp4 format or if the provided exercise type is not recognized.
        """

    if video_path is None or not video_path.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Invalid video format, please upload an .mp4 file.")

    if exercise_type not in ['squat', 'deadlift', 'benchpress']:
        raise HTTPException(status_code=400, detail="Unsupported exercise type")

    cap = cv2.VideoCapture(video_path)
    angles = []  # Initialize an empty list to store angles for deadlifts
    min_angle_squat = float('inf')  # Initialize with infinity for finding the minimum squat angle
    min_angle_benchpress = float('inf')  # Initialize with infinity for finding the minimum bench press angle
    final_angle = None  # Initialize final_angle

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    if exercise_type == 'squat':
                        angle = analyze_squat(landmarks)
                        min_angle_squat = min(min_angle_squat, angle)  # Update the minimum angle for squats
                    elif exercise_type == 'deadlift':
                        angle = analyze_deadlift(landmarks)
                        angles.append(angle)  # Accumulate angles for deadlifts
                    elif exercise_type == 'benchpress':
                        angle = analyze_benchpress(landmarks)
                        min_angle_benchpress = min(min_angle_benchpress, angle)

                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.putText(frame, f"Angle: {angle:.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

                cv2.imshow('Video Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during video processing: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Handle the final calculation based on the exercise type
        if exercise_type == 'squat':
            final_angle = min_angle_squat
        elif exercise_type == 'deadlift':
            final_angle = sum(angles) / len(angles) if angles else 0  # Calculate the average for deadlifts
        elif exercise_type == 'benchpress':
            final_angle = min_angle_benchpress

        return final_angle, exercise_type


def chat_with_ai_video(final_angle, exercise_type):
    """
       Generates a response from an AI model regarding the form of an exercise based on the calculated angle.

       This function takes the final angle calculated for a specific exercise and the type of exercise (squat, deadlift
       or bench press) as inputs. It constructs a prompt that describes the user's performance in terms of the relevant body angle.
       This prompt is then sent to an AI chat model, which generates a response providing feedback or guidance on how to improve exercise form.

       Parameters:
       - final_angle (float): The calculated angle relevant to the exercise form being analyzed.
       - exercise_type (str): The type of exercise being analyzed. Valid options are 'squat', 'deadlift', and 'benchpress'.

       Returns:
       - str: A string containing the AI model's generated response, offering feedback or tips on the user's exercise form.

       Raises:
       - ValueError: If an invalid exercise type is provided. Only 'squat', 'deadlift', and 'benchpress' are accepted.
       """

    if exercise_type == "squat":
        prompt = f"I am analyzing your squat form and it seems your knee angle is {final_angle} degrees. This is good for squatting!"
    elif exercise_type == "deadlift":
        prompt = f"I am analyzing your deadlift form and it seems your shoulder angle is {final_angle} degrees. Try to keep your back straighter."
    elif exercise_type == "benchpress":
        prompt = f"I am analyzing your bench press form and it seems your elbow angle is {final_angle} degrees. Make sure to keep your elbows stable."
    else:
        raise HTTPException(status_code=400, detail="Invalid exercise type")
    chat_completion = client.chat.completions.create(
        messages=
        [
            {
                "role": "system",
                "content": f"You are a professional powerlifting and strength training coach. "
                           f"Provide guidance on improving performance in {exercise_type}. "
                           "Focus on optimizing knee angles for squats, back angles for deadlifts, and elbow angles for bench presses based on the exercise type I gave you. "
                           "Use the following method to measure angles: a = np.array(a) (First point), b = np.array(b) (Mid point), c = np.array(c) (End point), "
                           "radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]), "
                           "angle = np.abs(radians*180.0/np.pi), if angle > 180.0: angle = 360 - angle. "
                           "Provide explanations and tips clearly."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content


def say_text(text):
    """
        Utilizes the pyttsx3 text-to-speech library to audibly speak the provided text.

        This function initializes the text-to-speech engine, submits the given text for vocalization, and waits for the
        speech process to complete before returning control to the caller. It effectively enables the application to
        provide audible feedback or instructions to the user.

        Parameters:
        - text (str): The text string to be spoken by the text-to-speech engine.

        Returns:
        None
        """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def process_video_real_time(duration: int, exercise_type: str):
    """
        Processes a real-time video stream from a webcam, analyzes exercise form, and provides feedback.

        This function captures video from the default camera (e.g., a webcam) and processes the video stream
        in real-time to analyze the form of a specified exercise. It utilizes MediaPipe for pose estimation
        to calculate specific angles related to the exercise being performed and provides visual feedback
        on the form directly on the video stream. The function also counts the number of properly completed
        exercises based on predetermined angle thresholds.

        Parameters:
        - duration (int): The duration in seconds for which the video stream should be processed.
        - exercise_type (str): The type of exercise to be analyzed. Valid options include 'squat', 'deadlift',
          and 'benchpress'.

        Returns:
        - int: The count of properly completed exercises based on the form analysis.
        """
    cap = cv2.VideoCapture(0)
    display_message_time = 0  # Initialize the timer for displaying the message
    count = 0  # Initialize count exercise
    exercise_completed = False  # Track whether a squat has been completed
    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            frame_bgr_for_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if exercise_type == 'squat':
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    knee_angle = calculate_angle(hip, knee, ankle)

                    # Display the knee angle and squat count
                    cv2.putText(frame_bgr_for_display, f"Knee Angle: {knee_angle:.2f}",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2, cv2.LINE_AA)

                    cv2.putText(frame_bgr_for_display, f"Squat Count: {count}",
                                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2, cv2.LINE_AA)

                    cv2.putText(frame_bgr_for_display, f"Time: {time.time() - start_time:.2f}",
                                (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2, cv2.LINE_AA)

                    if knee_angle < 90 and not exercise_completed:
                        display_message_time = time.time() + 5  # Display "GOOD SQUAT" message for 5 seconds
                        exercise_completed = True  # Mark squat as completed to avoid recounting during the same squat
                        threading.Thread(target=say_text, args=('Good Squat',)).start()

                    if knee_angle > 90 and exercise_completed:
                        count += 1  # Increment count when coming back up from a squat
                        exercise_completed = False  # Reset for the next squat

                    if time.time() < display_message_time:
                        cv2.putText(frame_bgr_for_display, "GOOD SQUAT",
                                    (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2, cv2.LINE_AA)

                    mp_drawing.draw_landmarks(frame_bgr_for_display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                cv2.imshow('Knee Angle Detection', frame_bgr_for_display)
                if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > duration:
                    break

            if exercise_type == 'deadlift':
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    back_angle = calculate_angle(shoulder, hip, knee)

                    # Display the back angle and deadlift count
                    cv2.putText(frame_bgr_for_display, f"Back Angle: {back_angle:.2f}",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2, cv2.LINE_AA)

                    cv2.putText(frame_bgr_for_display, f"Deadlift Count: {count}",
                                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2, cv2.LINE_AA)

                    cv2.putText(frame_bgr_for_display, f"Time: {time.time() - start_time:.2f}",
                                (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2, cv2.LINE_AA)

                    if back_angle < 60 and not exercise_completed:
                        display_message_time = time.time() + 5  # Display "GOOD DEADLIFT" message for 5 seconds
                        exercise_completed = True  # Mark deadlift as completed to avoid recounting during the same deadlift
                        threading.Thread(target=say_text, args=('Good Deadlift',)).start()

                    if back_angle > 60 and exercise_completed:
                        count += 1  # Increment count when coming back up from a deadlift
                        exercise_completed = False  # Reset for the next deadlift

                    if time.time() < display_message_time:
                        cv2.putText(frame_bgr_for_display, "GOOD DEADLIFT",
                                    (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2, cv2.LINE_AA)

                    mp_drawing.draw_landmarks(frame_bgr_for_display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                cv2.imshow('Back Angle Detection', frame_bgr_for_display)
                if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > duration:
                    break

            if exercise_type == 'benchpress':
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                    elbow_angle = calculate_angle(wrist, elbow, shoulder)

                    # Display the elbow angle and benchpress count
                    cv2.putText(frame_bgr_for_display, f"Elbow Angle: {elbow_angle:.2f}",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2, cv2.LINE_AA)

                    cv2.putText(frame_bgr_for_display, f"Benchpress Count: {count}",
                                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2, cv2.LINE_AA)

                    cv2.putText(frame_bgr_for_display, f"Time: {time.time() - start_time:.2f}",
                                (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2, cv2.LINE_AA)

                    if elbow_angle < 75 and not exercise_completed:
                        display_message_time = time.time() + 5  # Display "GOOD BENCHPRESS" message for 5 seconds
                        exercise_completed = True  # Mark benchpress as completed to avoid recounting during the same benchpress
                        threading.Thread(target=say_text, args=('Good Benchpress',)).start()

                    if elbow_angle > 75 and exercise_completed:
                        count += 1  # Increment count when coming back up from a benchpress
                        exercise_completed = False  # Reset for the next benchpress

                    if time.time() < display_message_time:
                        cv2.putText(frame_bgr_for_display, "GOOD BENCHPRESS",
                                    (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2, cv2.LINE_AA)

                    mp_drawing.draw_landmarks(frame_bgr_for_display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                cv2.imshow('Elbow Angle Detection', frame_bgr_for_display)
                if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > duration:
                    break

        cap.release()
        cv2.destroyAllWindows()
        return count


def email_sender(email, feedback_content):
    """
        Sends an email to the specified address with feedback on the user's exercise.

        This function constructs an email with feedback content and sends it to the given email address.
        It uses an SMTP service (specified by its URL and credentials stored in environment variables)
        to send the email. The function checks the response from the SMTP service to determine if the
        email was sent successfully.

        Parameters:
        - email (str): The email address to which the feedback will be sent.
        - feedback_content (str): The content of the feedback to be included in the email body.

        Returns:
        - dict: A dictionary with a 'message' key indicating the result of the email sending attempt.
                The message can either confirm successful sending or indicate failure.
        """
    url = "https://smtpjs.com/v3/smtpjs.aspx"
    payload = {
        "Host": "smtp.elasticemail.com",
        "Username": os.getenv("USERNAME_EMAIL"),
        "Password": os.getenv("PASSWORD_EMAIL"),
        "To": email,
        "From": "itay.golan01@post.runi.ac.il",
        "Subject": 'Your Feedback on your exercise',
        "Body": "Your feedback on your exercise is: " + feedback_content,
        "Action": "Send"
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return {'message': 'Email sent successfully'}
    else:
        return {'message': 'Failed to send email'}


def chat_with_ai_video_real_time(good_count: int, duration: int, exercise_type: str):
    """
            Interacts with an AI model in real time to get feedback on exercise form based on the count of properly executed exercises.

            This function constructs a chat prompt describing the user's performance on a specific exercise type during a real-time
            session. It then sends this prompt to an AI model to generate feedback or suggestions for improvement. The AI model's
            response is based on the number of good repetitions the user has completed within a specified time frame.

            Parameters:
            - good_count (int): The number of good repetitions of the exercise completed by the user.
            - duration (int): The total duration of the exercise session in seconds.
            - exercise_type (str): The type of exercise performed (e.g., "squat", "deadlift", "benchpress").

            Returns:
            - str: The feedback or suggestions generated by the AI model based on the user's performance.
            """
    if exercise_type == "squat":
        angle = "knee"
    elif exercise_type == "deadlift":
        angle = "back"
    elif exercise_type == "benchpress":
        angle = "elbow"
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional powerlifting and strength training coach."
                           f"Provide guidance on improving performance in {exercise_type}."
                           f"Focus on optimizing {angle} angles and overall form for effective and safe {exercise_type}s."
            },
            {
                "role": "user",
                "content": f"Hi, I've been working on my {exercise_type} form and would like some tips to improve."
                           f"I managed to complete a certain number of good {exercise_type}s within a given time frame."
                           "Can you provide some feedback and suggestions for improvement?"
            },
            {
                "role": "user",
                "content": f"I completed {good_count} good {exercise_type}s in {duration} seconds."
                           f"Based on this, can you give me some specific advice or exercises to help improve my {exercise_type} form and performance?"
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content


@app.post("/Video Processing")
async def process_video_If_you_dont_upload_anything_it_will_analyze_video_in_real_time(Video: UploadFile = File(None), Exercise_type: str = Form(...),
                                 Duration_in_real_time: Optional[int] = Form(0), email: EmailStr = Form(...)):
    """
        Processes an uploaded exercise video for form analysis or conducts real-time exercise form analysis,
        then provides feedback based on AI analysis.

        This endpoint serves two functions based on the input:
        1. If a video is uploaded and an exercise type is specified, it analyzes the form in the video, provides feedback,
           and optionally sends this feedback via email.
        2. If no video is uploaded but a duration and exercise type are specified, it initiates a real-time exercise form analysis
           for the specified duration and exercise type, then provides feedback and optionally sends this feedback via email.

        Parameters:
        - Video (UploadFile, optional): The video file uploaded by the user for form analysis. Defaults to None.
        - Exercise_type (str): The type of exercise to be analyzed. Must be one of 'squat', 'deadlift', or 'benchpress'.
        - Duration_in_real_time (int, optional): The duration in seconds for real-time exercise analysis. Defaults to 0.
        - email (EmailStr): The email address to which feedback should be sent.

        Returns:
        - A dictionary containing either the feedback from the video analysis or the result of the real-time analysis,
          along with a success message.

        Raises:
        - HTTPException: If the exercise type is not one of the specified valid types or other input validations fail.
        """

    if Exercise_type and Exercise_type.lower() not in ["squat", "deadlift", "benchpress"]:
        raise HTTPException(status_code=400, detail="Invalid exercise type. Please choose from 'squat', 'deadlift', or 'benchpress'.")

    if Video and Exercise_type:
        temp_video_path = f"temp_{Video.filename}"
        with open(temp_video_path, 'wb') as buffer:
            shutil.copyfileobj(Video.file, buffer)
        final_angle, exercise = analyze_exercise_form(temp_video_path, Exercise_type.lower())
        feedback = chat_with_ai_video(final_angle, exercise)

        os.remove(temp_video_path)  # Cleanup

        if email:
            email_sender(email, feedback)

        return {"feedback": feedback, "message": "Uploaded video analysis complete."}

    elif Duration_in_real_time and Duration_in_real_time > 0:
        good_count = process_video_real_time(Duration_in_real_time, Exercise_type.lower())
        ai_feedback = chat_with_ai_video_real_time(good_count, Duration_in_real_time, Exercise_type.lower())

        if email:
            email_sender(email, ai_feedback)

        return {"message": "email sent successfully", "feedback": ai_feedback}

    else:
        return {
            "message": "Please specify a positive duration for real-time processing or upload a video with an exercise type for analysis."}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
