# get the number of lines in a text
def count_lines(text):
    return text.count('\n') + 1

# Test cases
print(count_lines("Hello\nWorld"))  # Expected output: 2
print(count_lines("Hello World"))  # Expected output: 1
print(count_lines("Hello\n\n\nWorld"))  # Expected output: 4
print(count_lines(""))  # Expected output: 1
text= "Hello World\nHello"
print(text.split("\n"))