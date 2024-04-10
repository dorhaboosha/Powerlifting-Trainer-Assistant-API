# crossfit_trainning_api- made by Dor Habooshe, Itay Golan and Moran Herzlinger
Link to the video- https://youtu.be/xNZTbHj9JV0

Our project is designed for Powerlifting trainers which will help them in their training routine.

There is a registration function ("/Registration") where the user must register their email, height, weight, and name. Once the user is registered, we save his data which includes the data the user gave and calculate his BMI and create 3 empty lists for the record weights in the exercises that the user will do in a database.

There is a search function ("/Show User") for the user that we enter an email and if this email exists in a database, then we will see all his data and if it does not exist then we will receive a message that the user does not exist.

There is an update function ("/Update Record Weight") where we basically update the records of the exerciser, that is the record of this exerciser when he adds weight to the exercise and as the weight increases so does his record.

In the last function ("/Video Processing") the trainer can upload a video of an exercise and write the type of exercise (between 3 options of the powerlifting exercise squat, deadlift, bench-press )he did and we analyze this video and take the minimum angle he reached (in every exercise the angle is different, that is, in the squat we check the knee angle, in the deadlift we check the back angle and bench press we check elbow angle) and send this angle to openAI and receive feedback on the video if the trainer did the exercise well or not and tips for doing the exercise well. There is also an option to do the exercise in real time, so we simply don't upload a video and record the type of exercise and how long he is going to do it. During this time, we see the trainee doing the exercise while on the photo screen we have the time and the size of the angle we are checking in the exercise. If according to our analysis the exercise has reached the angle where the exercise is good, then there is a voice that says the exercise is good. Finally, we check how many times he did the exercise well and then ask the openAI for feedback to the trainee.

We send the feedback to the email that the trainee brings us and show it on the screen.
