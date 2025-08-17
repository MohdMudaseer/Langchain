from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Person(BaseModel):
    name:str="Mudaseer"
    age:Optional[int]=None
    email:Optional[EmailStr]=None
    cgpa:Optional[float]=Field(None,gt=0,lt=10.0)

student={"name":"Nehal",'age':"21","email":"abc@gmail.com","cgpa":8.5}
person=Person(**student)
print(person)

person_dict=dict(person)
print(person_dict)

person_json=person.model_dump_json()
print(person_json)
