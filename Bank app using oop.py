# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:49:52 2025

@author: user
"""

import math

class OnlineBankingApp:
    def __init__(self):
        self.name = ""
        self.pin = ""
        self.cb = 0

    def signin(self):
        print("Please create your account.")
        self.name = input("Please create your username: ")
        while True:
            self.pin = input("Please create your 6-digit pin: ")
            if len(self.pin) == 6:
                break
            else:
                print("The pin has to be 6 digits.")
        print("Thank you for creating your bank account.")

    def forgotpin(self):
        while True:
            recoverpin = input("Please create your new 6-digit pin: ")
            if len(recoverpin) == 6:
                self.pin = recoverpin
                print("New pin reset successfully. Please log in.")
                self.login()
                break
            else:
                print("The pin has to be 6 digits.")

    @staticmethod
    def depositinterest(p, r, t):
        # A = P * e^(rt), the formula for compound interest
        p = float(p)
        r = float(r)
        t = float(t)
        rt = r * t
        e = math.exp(rt)
        return p * e

    def exit_app(self):
        answer = input("Do you want to conduct another transaction? Yes or No: ").strip().lower()
        if answer == "yes":
            self.login()
        elif answer == "no":
            print("Thank you for using this app.")
        else:
            print("Option not available.")
            self.mainmenu()

    def login(self):
        name1 = input("Please enter your username: ")
        pin1 = input("Please enter your pin: ")

        if name1 == self.name and pin1 == self.pin:
            print(f"Welcome to the online banking application, {self.name}")
            while True:
                print("Please choose from the menu below:")
                menu = [
                    "1-Deposit",
                    "2-Withdraw",
                    "3-Transfer",
                    "4-Check Balance",
                    "5-Deposit interest rate",
                    "6-Calculate compound interest",
                    "7-End Session"
                ]
                for item in menu:
                    print(item)

                try:
                    choose = int(input("Please enter the number of your choice: "))
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue

                if choose == 1:
                    try:
                        d = float(input("Enter amount to deposit: "))
                        self.cb += d
                        print(f"Your current balance is: {self.cb}")
                    except ValueError:
                        print("Invalid amount.")

                elif choose == 2:
                    try:
                        w = float(input("Enter amount to withdraw: "))
                        if w > self.cb:
                            print("Insufficient funds for this transaction.")
                        else:
                            self.cb -= w
                            print(f"{w} has been withdrawn from your account. Your current balance is: {self.cb}")
                    except ValueError:
                        print("Invalid amount.")

                elif choose == 3:
                    destination = input("Please enter the receiver's 10-digit account number: ")
                    if len(destination) == 10:
                        try:
                            amount = float(input("Enter amount to transfer: "))
                            if amount > self.cb:
                                print("Insufficient funds for this transaction.")
                            else:
                                self.cb -= amount
                                print(f"A transaction of {amount} has been transferred to {destination}. Your current balance is: {self.cb}")
                        except ValueError:
                            print("Invalid amount.")
                    else:
                        print("Invalid receiver's account number. It must be 10 digits.")

                elif choose == 4:
                    print(f"Your current balance is: {self.cb}")

                elif choose == 5:
                    if self.cb > 50000:
                        rate = 3
                    elif self.cb > 30000:
                        rate = 2
                    else:
                        rate = 1.5
                    print(f"Current deposit interest rate is: {rate}%")

                elif choose == 6:
                    options = {
                        "1": "Calculate compound interest based on current balance",
                        "2": "Calculate compound interest based on a new deposit input",
                    }
                    for key, value in options.items():
                        print(f"{key}-{value}")

                    choice = input("Enter your choice: ")

                    if choice == "1":
                        try:
                            timing = float(input("Enter preferred investment duration (in years): "))
                            if self.cb > 50000:
                                ratex = 3 / 100
                            elif self.cb > 30000:
                                ratex = 2 / 100
                            else:
                                ratex = 1.5 / 100
                            print(f"Your balance after {timing} years will be: {self.depositinterest(self.cb, ratex, timing)}")
                        except ValueError:
                            print("Invalid input.")

                    elif choice == "2":
                        try:
                            timing = float(input("Enter preferred investment duration (in years): "))
                            amount_to_invest = float(input("Enter amount to invest: "))
                            if self.cb > 50000:
                                ratex = 3 / 100
                            elif self.cb > 30000:
                                ratex = 2 / 100
                            else:
                                ratex = 1.5 / 100
                            print(f"The invested amount after {timing} years will be: {self.depositinterest(amount_to_invest, ratex, timing)}")
                        except ValueError:
                            print("Invalid input.")

                    else:
                        print("Invalid option.")

                elif choose == 7:
                    self.exit_app()
                    break

                else:
                    print("Option not available.")

        else:
            print("Either Username or Pin is incorrect. Do you have an existing account?")
            list1 = ["1 = Yes", "2 = No"]
            for item in list1:
                print(item)
            try:
                inp = int(input("Enter your choice: "))
                if inp == 1:
                    list2 = ["1-Attempt login again", "2-Forgot pin"]
                    for option in list2:
                        print(option)
                    theanswer = int(input("Enter your choice: "))
                    if theanswer == 1:
                        self.login()
                    elif theanswer == 2:
                        self.forgotpin()
                    else:
                        print("Option not available.")
                        self.login()
                elif inp == 2:
                    print("Please create an account.")
                    self.signin()
                else:
                    print("Option not available.")
            except ValueError:
                print("Invalid input.")

    def mainmenu(self):
        while True:
            try:
                optionone = int(input("Choose 1 to Sign in, Choose 2 to Login if you already have an account: "))
                if optionone == 1:
                    self.signin()
                    break
                elif optionone == 2:
                    self.login()
                    break
                else:
                    print("Option not available.")
            except ValueError:
                print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    app = OnlineBankingApp()
    app.mainmenu()
