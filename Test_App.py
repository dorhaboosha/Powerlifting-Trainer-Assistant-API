from dotenv import load_dotenv
from fastapi.testclient import TestClient
import pytest
import sqlite3
from main import app, get_db_connection, calculate_bmi, analyze_squat, analyze_exercise_form, \
    analyze_benchpress, analyze_deadlift, mp_pose, chat_with_ai_video, say_text, email_sender, calculate_angle, update_weight
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
import main
import os.path
import json

load_dotenv()

client = TestClient(app)

@pytest.fixture
def mock_db_connection(mocker):
    mocker.patch('sqlite3.connect', return_value=MagicMock(sqlite3.Connection))


def test_get_db_connection(mock_db_connection):
    conn = get_db_connection()
    assert conn is not None

def test_calculate_bmi():
    height = 1.75  # meters
    weight = 75  # kg
    expected_bmi = 24.49
    bmi = calculate_bmi(height, weight)
    assert bmi == expected_bmi

def test_register_user():
    pre_cleanup()

    response = client.post("/Registration", json={
        "email": "test@example.com",
        "height": 1.75,
        "weight": 75,
        "name": "Test User"
    })
    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to our program"

    post_cleanup()

def post_cleanup():
    """Delete the test user after the test."""
    conn = get_db_connection()
    conn.execute('DELETE FROM Users WHERE email = ?', ("test@example.com",))
    conn.commit()
    conn.close()

def pre_cleanup():
    """Ensure the test user does not exist before running the test."""
    conn = get_db_connection()
    conn.execute('DELETE FROM Users WHERE email = ?', ("test@example.com",))
    conn.commit()
    conn.close()


def test_get_user():
    test_email = "NotExists@example.com"
    response = client.get(f"/users/?user_email={test_email}")
    assert response.status_code == 404  # Assuming the user does not exist in the database for this test


def test_calculate_angle_degrees():
    # Points forming a 45-degree angle
    a = [0, 0]
    b = [1, 0]
    c = [1, 1]
    expected_angle = 90.0

    calculated_angle = calculate_angle(a, b, c)

    assert calculated_angle == expected_angle

# Helper function to mock landmarks based on given coordinates
def mock_landmark(x, y):
    mock_landmark = MagicMock()
    mock_landmark.x = x
    mock_landmark.y = y
    return mock_landmark

# Define tests for analyze_squat
def test_analyze_squat():
    # Mocking landmarks for the squat position
    hip = mock_landmark(0.5, 0.5)
    knee = mock_landmark(0.5, 0.6)
    ankle = mock_landmark(0.5, 0.7)

    landmarks = MagicMock()
    landmarks.landmark = [None] * 33  # Assuming 33 landmarks for simplicity
    landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value] = hip
    landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value] = knee
    landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value] = ankle

    expected_angle = calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
    calculated_angle = analyze_squat(landmarks)

    assert calculated_angle == pytest.approx(expected_angle)

# Define tests for analyze_deadlift
def test_analyze_deadlift():
    # Mocking landmarks for the deadlift position
    shoulder = mock_landmark(0.5, 0.4)
    hip = mock_landmark(0.5, 0.5)
    knee = mock_landmark(0.5, 0.6)

    landmarks = MagicMock()
    landmarks.landmark = [None] * 33
    landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = shoulder
    landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value] = hip
    landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value] = knee

    expected_angle = calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))
    calculated_angle = analyze_deadlift(landmarks)

    assert calculated_angle == pytest.approx(expected_angle)

# Define tests for analyze_benchpress
def test_analyze_benchpress():
    # Mocking landmarks for the bench press position
    wrist = mock_landmark(0.4, 0.5)
    elbow = mock_landmark(0.5, 0.5)
    shoulder = mock_landmark(0.6, 0.5)

    landmarks = MagicMock()
    landmarks.landmark = [None] * 33
    landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value] = wrist
    landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value] = elbow
    landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = shoulder

    expected_angle = calculate_angle((wrist.x, wrist.y), (elbow.x, elbow.y), (shoulder.x, shoulder.y))
    calculated_angle = analyze_benchpress(landmarks)

    assert calculated_angle == pytest.approx(expected_angle)


@pytest.fixture
def mock_db_connection(mocker):
    mocker.patch('sqlite3.connect', return_value=MagicMock(sqlite3.Connection))

@pytest.fixture
def mock_openai_client(mocker):
    mocker.patch('main.client.chat.completions.create', return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Mocked AI response"))]))

def test_analyze_exercise_form_invalid_input(mock_openai_client):
    # Test invalid video path
    with pytest.raises(HTTPException) as e:
        analyze_exercise_form(None, 'squat')
    assert e.value.status_code == 400
    assert "Invalid video format" in str(e.value.detail)

    # Test unsupported exercise type
    with pytest.raises(HTTPException) as e:
        analyze_exercise_form("video.mp4", "unsupported_exercise")

    # Test for all exercise types
    for exercise_type in ['squat', 'deadlift', 'benchpress']:
        angle, analyzed_type = analyze_exercise_form("video.mp4", exercise_type)
        assert angle is not None
        assert analyzed_type == exercise_type
        if exercise_type == 'squat':
            assert isinstance(angle, (float, int))

        elif exercise_type == 'deadlift':
            assert isinstance(angle, (float, int))


        elif exercise_type == 'benchpress':
            assert isinstance(angle, (float, int))



def test_chat_with_ai_video():
    # Mock final angle and exercise type
    final_angle = 45  # Example final angle
    exercise_type = "squat"  # Example exercise type

    response = chat_with_ai_video(final_angle, exercise_type)

    # Assert
    assert "45 degrees" in response
    assert "squat" in response
    assert "good" in response


@patch('pyttsx3.init')
def test_say_text(mock_init):
    # Define the text to be spoken
    text = "This is a test message."
    say_text(text)
    mock_init.assert_called_once()
    mock_init.return_value.say.assert_called_once_with(text)
    mock_init.return_value.runAndWait.assert_called_once()


@patch('main.mp_drawing.draw_landmarks')  # Mock draw_landmarks
@patch('main.cv2.putText')  # Mock cv2.putText
@patch('main.cv2.cvtColor')  # Mock cv2.cvtColor
@patch('main.cv2.VideoCapture')
@patch('main.cv2.imshow')
@patch('main.cv2.waitKey')
@patch('main.cv2.destroyAllWindows')
@patch('main.mp_pose.Pose')
@patch('main.calculate_angle')
def test_process_video_real_time(mock_calculate_angle, mock_Pose, mock_destroyAllWindows, mock_waitKey, mock_imshow, mock_VideoCapture, mock_cvtColor, mock_putText, mock_draw_landmarks):
    # Set up mock for calculate_angle to return a scalar value
    mock_calculate_angle.return_value = 45.0  # Example angle

    mock_putText.side_effect = lambda *args, **kwargs: None
    mock_draw_landmarks.side_effect = lambda *args, **kwargs: None

    mock_VideoCapture_instance = mock_VideoCapture.return_value
    mock_VideoCapture_instance.isOpened.return_value = True
    mock_frame_bgr = MagicMock()
    mock_frame_bgr.shape = (100, 100, 3)
    mock_VideoCapture_instance.read.return_value = (True, mock_frame_bgr)

    mock_results = MagicMock()
    mock_results.pose_landmarks.landmark = [MagicMock() for _ in range(33)]
    mock_Pose.return_value.process.return_value = mock_results
    mock_Pose.return_value.POSE_CONNECTIONS = [(15, 21)]

    duration = 10
    exercise_type = 'squat'
    count = main.process_video_real_time(duration, exercise_type)

    # Assertions
    assert count == 0
    assert mock_VideoCapture_instance.isOpened.called
    assert mock_VideoCapture_instance.read.called
    assert mock_Pose.called
    assert mock_destroyAllWindows.called
    assert mock_cvtColor.called
    assert mock_putText.called
    assert mock_draw_landmarks.called




@patch('requests.post')
def test_email_sender_success(mock_post):
    # Mock the response from requests.post to simulate a successful email sending
    mock_post.return_value.status_code = 200

    email = 'test@example.com'
    feedback_content = 'This is a test feedback'
    result = email_sender(email, feedback_content)

    # Check if the email was sent successfully
    assert result == {'message': 'Email sent successfully'}

    # Check if requests.post was called with the correct arguments
    expected_payload = {
        "Host": "smtp.elasticemail.com",
        "Username": os.getenv("USERNAME_EMAIL"),
        "Password": os.getenv("PASSWORD_EMAIL"),
        "To": email,
        "From": "itay.golan01@post.runi.ac.il",
        "Subject": 'Your Feedback on your exercise',
        "Body": "Your feedback on your exercise is: " + feedback_content,
        "Action": "Send"
    }
    mock_post.assert_called_once_with("https://smtpjs.com/v3/smtpjs.aspx", json=expected_payload)

    @pytest.mark.asyncio
    async def test_update_weight_success(mocker):
        # Mocking database connection and execute method
        conn_mock = mocker.MagicMock()
        conn_execute_mock = mocker.MagicMock()
        conn_mock.execute.return_value = conn_execute_mock
        mocker.patch('main.get_db_connection', return_value=conn_mock)

        # Mocking user data in the database
        email = "test@example.com"
        user_data = {"email": email, "squat": "[]", "bench_press": "[]", "deadlift": "[]"}
        conn_execute_mock.fetchone.return_value = user_data

        # New weight values
        new_squat = 100.0
        new_bench_press = 150.0
        new_deadlift = 200.0

        # Call the function and await the result
        response = await update_weight(email=email, new_squat=new_squat, new_bench_press=new_bench_press,
                                       new_deadlift=new_deadlift)

        # Assert the response
        assert response == {"message": "User's records updated successfully"}

        # Assert that the database was queried and updated correctly
        conn_mock.execute.assert_called_once_with(
            'UPDATE Users SET squat = ?, deadlift = ?, bench_press = ? WHERE email = ?',
            (json.dumps([new_squat]), json.dumps([new_deadlift]), json.dumps([new_bench_press]), email)
        )
        conn_execute_mock.commit.assert_called_once()
        conn_execute_mock.close.assert_called_once()