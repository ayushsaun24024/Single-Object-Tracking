# Object-Oriented Programming in Python
## Tutorial 13: OOP Basics

---

## 1. What is a Data Type?

A data type is a classification of data that defines:
- The type of value a variable can hold
- The operations that can be performed on that value
- The way the data is stored in memory

### Examples of Primitive Data Types
- Integer: Whole numbers (e.g., 42, -17)
- Float: Decimal numbers (e.g., 3.14, -0.5)
- String: Text data (e.g., "Hello, World!")
- Boolean: True or False values

### Custom Data Types
- Allow creating more complex, user-defined data structures
- Represent real-world entities with their properties and behaviors

---

## 2. Introduction to Object-Oriented Programming (OOP)

### What is OOP?
- A programming paradigm based on the concept of "objects"
- Objects contain data (attributes) and code (methods)
- Focuses on creating reusable, modular code
- Helps in organizing and structuring complex software systems

### Key Principles of OOP
1. **Encapsulation**: Bundling data and methods that operate on that data
2. **Inheritance**: Creating new classes based on existing classes
3. **Polymorphism**: Ability of objects to take on multiple forms
4. **Abstraction**: Hiding complex implementation details

---

## 3. Defining a Class in Python

### What is a Class?
- A blueprint for creating objects
- Defines attributes and methods that the objects will have
- Serves as a template for creating instances

### Basic Class Structure
```python
class ClassName:
    # Class attributes (shared by all instances)
    class_attribute = value

    # Constructor method
    def __init__(self, param1, param2):
        # Instance attributes (unique to each object)
        self.attribute1 = param1
        self.attribute2 = param2

    # Instance methods
    def method_name(self, parameters):
        # Method implementation
        return something
```

---

## 4. Complex Number Class Example

### Implementing a Complex Number Class
```python
class Complex:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    # Method to compute modulus
    def modulus(self):
        return (self.real**2 + self.imag**2)**0.5

    # Method to multiply complex numbers
    def multiply(self, other):
        new_real = self.real * other.real - self.imag * other.imag
        new_imag = self.real * other.imag + self.imag * other.real
        return Complex(new_real, new_imag)

    # Dunder method for equality comparison
    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag

    # String representation method
    def __str__(self):
        return f"{self.real} + {self.imag}i"

# Usage example
c1 = Complex(3, 4)
c2 = Complex(1, 2)
print(f"Modulus of c1: {c1.modulus()}")
print(f"c1 * c2: {c1.multiply(c2)}")
```

---

## 5. Stack Class for String Palindrome

### Implementing a Stack
```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")

    def is_empty(self):
        return len(self.items) == 0

# Palindrome checker function
def is_palindrome(s):
    # Remove spaces and convert to lowercase
    s = s.replace(" ", "").lower()
    
    # Create a stack
    stack = Stack()
    
    # Push all characters onto the stack
    for char in s:
        stack.push(char)
    
    # Compare original string with reversed stack
    reversed_string = ""
    while not stack.is_empty():
        reversed_string += stack.pop()
    
    return s == reversed_string

# Test palindrome
print(is_palindrome("race a car"))  # True
print(is_palindrome("hello"))       # False
```

---

## 6. Best Practices in OOP

1. **Keep classes focused**: Each class should have a single, well-defined responsibility
2. **Use meaningful names**: Class and method names should clearly describe their purpose
3. **Minimize complexity**: Break down complex classes into smaller, manageable parts
4. **Prefer composition over inheritance**: Use object composition when possible
5. **Follow Python naming conventions**:
   - CamelCase for class names
   - snake_case for methods and attributes

---

## 7. Recommended Learning Path

1. Practice creating simple classes
2. Implement real-world scenarios using classes
3. Explore inheritance and polymorphism
4. Study design patterns
5. Read open-source Python projects

### Additional Resources
- Python's official documentation
- "Python Crash Course" by Eric Matthes
- "Fluent Python" by Luciano Ramalho
