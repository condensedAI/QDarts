import tkinter as tk

def on_button_click():
    user_input = entry.get()
    label.config(text=f"Hello, {user_input}!")

# Create the main window
root = tk.Tk()
root.title("Greeting Application")

# Create an entry widget
entry = tk.Entry(root)
entry.grid(row=0, column=0, padx=10, pady=10)

# Create a button widget
button = tk.Button(root, text="Greet", command=on_button_click)
button.grid(row=0, column=1, padx=10, pady=10)

# Create a label widget
label = tk.Label(root, text="Enter your name and press Greet")
label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Run the application
root.mainloop()