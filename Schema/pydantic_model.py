from pydantic import BaseModel,Field
from typing import Annotated

class UserInput(BaseModel):
    age:Annotated[float,Field(...,gt=0,description="Age of the student in years",examples=[18])]
    gender:Annotated[float,Field(...,description="Enter the Gender of the student: 0 for Female, 1 for Male",examples=["Female=0, Male=1"])]
    previous_grades:Annotated[float,Field(...,ge=0,le=101,description="Previous grades of the student out of 100",examples=[85])]
    attendance_percentage:Annotated[float,Field(...,ge=0,le=101,description="Attendance percentage of the student",examples=[90])]
    study_hours_per_week:Annotated[float,Field(...,gt=0,description="Number of study hours per week",examples=[15])]
    assignments_submitted:Annotated[float,Field(...,ge=0,le=101,description="Percentage of assignments submitted by the student",examples=[95])]
    exam_scores:Annotated[float,Field(...,ge=0,le=101,description="Exam scores of the student out of 100",examples=[88])]