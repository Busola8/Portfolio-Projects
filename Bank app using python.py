# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:48:06 2025

@author: user
"""

import math
import re
import string
print("Welcome to your online banking application")
def signin():
    global name #username
    global email
    global pin #password
    global cb #current balance
    print("Names must contain at least\nOne Upper case character\nOne lower case character\nA number\n")
    letters = string.ascii_letters
    numbers = string.digits
    while True:
       
        name = str(input("Please create your username "))
        def uppercase(name):
            for i in name:
                if i.isupper():
                    return True
            return False
        
        def lowercase(name):
            for i in name:
                if i.islower():
                    return True
            return False
                
        def number(name):
            for i in name:
                if i in numbers:
                    return True 
            return False
        
        
        if uppercase(name) != True:
            print("You need at least one Upper case character")
        if lowercase(name) != True:
            print("You need at least one lowercase character")  
        if number(name) != True:
            print("You need at least one digit") 
        else:
            break
    while True:
        pin = str(input("Please create your 6 digit pin "))
        if len(pin) == 6:
            pin = pin
            break
        else:
            print("The pin has to be 6 digits ")
            newpin = str(input("Please create your 6 digits pin "))
            if len(newpin) !=6:
                print("The pin hs to be 6 digits ")
                signin()
            else: 
                pin = newpin
                break
    while True:
        email = input("Please enter your email address").strip()
        if re.search(r"^\w+@(\w+\.)?\w+\.com$", email, re.IGNORECASE):
            print("Valid")
            print("Thank you for creating your bank account")
            break
        else:
            print("Invalid") 
#forgot pin
def forgotpin():
    while True:
        email1 =  input("Please enter your email address").strip()
        if re.search(r"^\w+@(\w+\.)?\w+\.com$", email1, re.IGNORECASE):
            break
        else:
            print("Invalid")
    if email1 == email:
        while True:
            recoverpin = str(input("Please create your new 6 digit pin "))
            if len(recoverpin) !=6:
                print("The pin hs to be 6 digits ")
                forgotpin()
            else:
                print("New pin reset successfully, Please log in ")
                pin = recoverpin
                login()
    else:
        print("Email does not match")
        forgotpin()
def depositinterest(p,r,t):
    #A= Pe**(rt) whch is the formula for compounf intrest
    p = float(p)
    r = float(r)
    t = float(t)
    rt = r*t
    e = math.exp(rt)
    A = p *e #future value on investment
    return A

def login():
    #name1 represents username
    #pin1 represents user pin
    name1 = str(input("Please enter your username here: "))
    pin1 = str(input("Please enter your pin: "))
    #check if it matches the created username and passwrod
    if name1 == name and pin1 == pin:
        print(f'Welcome to the online banking application,{name}')
        while True:
            print("Please choose the menu down here: ")
            listmenu = ("1-Deposit", "2-Withdraw", "3-Transfer", "4-Check Balance", "5-Deposit interest rate","6-Calculate compound interest", "7-End Session")
            for b in listmenu:
                print(b)
            choose = int(input("Please enter the number of your choice: "))
            d = 0 #d represents deposit
            w = 0 #w represents withrawal
            cb = 0 #represents current balance
            if choose == 1:
                d = int((input("Enter amount to deposit")))
                cb = d
                print("Your current balnce is"+" " +str(cb))
            elif choose == 2:
                w = int(input("Amount to withdraw: "))
                if w > cb:
                    print("Insufficient funds for this transaction")
                    login()
                else:
                    cb = cb-w
                    print(str(w) +" "+"has been withdrawn from your account,\nYour current balance is "+" "+ str(cb))
                    
            elif choose == 3:
                destination = str(input("Please enter reciever's 10 digit account number: "))
                if len(destination) == 10:
                    amount = int(input("Please enter amount to transfer: "))
                    if amount > cb:
                        print("Insufficient funds for this transaction")
                        login()
                    else: 
                        cb = cb - amount
                        print("A transaction of" + " "+ str(amount)+ " " + "has been transfered to" +" " + str(destination) + "Your current balance is"+ str(cb))
                else:
                    print("Invalid reciever's account number, Account number must be 10 digits. ")
                    login()
                    
            elif choose == 4:
                print("Your current balance is " +" "+ str(cb))
                
            elif choose == 5:
                if d > 50000:
                    rate = 3
                elif d > 30000:
                    rate = 2
                else:
                    rate = 1.5
                print("Current deposit interet rate is"+ " "+str(rate)+"% ")
                
            elif choose == 6:
                listoption = {"1-Calculate compund interest based on current balance","2-Calculate the compound interest based on a new deposit input "}
                for n in listoption:
                    print(n)
                choices = int(input("Please enter your choice from the options above "))
                if choices == 1:
                    timing = str(input("Prefered investment duration "))
                    if d > 50000:
                        ratex = 3/100
                    elif d > 30000:
                        ratex = 2/100
                    else:
                        1.5/100
                    print("Your current balance"+ " "+ "timing"+ " "+ "years will be ")
                    print(depositinterest(cb, ratex, timing))
                elif choices == 2:
                    timing1 = str(input("Prefered investment duration "))
                    amount_to_invest = str(input("Please enter amount to invest "))
                    amount_to_invest = int(amount_to_invest)
                    if d > 50000:
                        ratex = 3/100
                    elif d > 30000:
                        ratex = 2/100
                    else:
                        1.5/100
                    print("Your current balance"+ " "+ "timing"+ " "+ "years will be")
                    print(depositinterest(amount_to_invest, ratex, timing))
            elif choose == 7:
                answer = str(input("Do you want to conduct another transaction? press 1 for Yes or 2 for No "))
                if answer == "1":
                    login()
                elif answer == "2":
                    print("Thank you for using this app")
                    break
                else:
                    print("Option not available")
                    mainmenu()
                
                
            else:
                print("Option is not availabe")
                login()
        
    else:
        print("Either Usernme or Pin is wrong, Do you have an existing account? ")
        list1 = ['1 = yes','2 = no']
        for i in list1:
            print(i)
        inp = int(input("Enter your choice below "))
        if inp == 1:
            list2 = ["1.Attempt login again","2. Forgot pin "]
            for e in list2:
                print(e)
            theanswer = str(input("Please enter your choice "))
            theanswer = int(theanswer)
            if theanswer == 1:
                login()
            elif theanswer == 2:
                forgotpin()
            else: 
                print('Option not available')
                login()
        elif inp == 2:
            print("Please create an accout")
            signin()
            
def mainmenu():
    optionone = int(input("Choose 1 to Sign in, Choose 2 to Login if you already have an account, Choose 3 to exit this application: "))
    if optionone == 1:
        signin()
    elif optionone == 2:
        login()
    elif optionone == 3:
        exit()
    else:
        print("Option not available ")
        mainmenu()
    exit()
    
def exit():
    answer = str(input("Do you want to conduct another transaction? Yes or No "))
    if answer == "Yes":
        login()
    elif answer == "No":
        print("Thank you for using this app")
        login()
    else:
        print("Option not available")
        mainmenu()
        
mainmenu()      
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        