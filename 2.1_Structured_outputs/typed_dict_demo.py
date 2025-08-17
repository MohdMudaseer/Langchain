from typing import _TypedDict

class Person(_TypedDict):
    name: str
    age: int
    email: str

ob=Person(name="Alice", age='30', email="alice@example.com")
print(ob)