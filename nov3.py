'''f=open("New.txt","w")
f.write("Hello World")
f.close()
import os
if os.path.exists("New.txt"):
    print("The file exists")
else:
    print("The file does not exist")'''
'''f=open("New.txt","w+")
f.write("Hello World")
f.write("\nWelcome to File Handling")
f.write("\nThis is the third line")

f=open("New.txt","a+")
f.write("\nThis is the fourth line")
f.seek(0)
print(f.read())
f.close()
'''
'''with open("New.txt","r") as f:
    c=f.read()
    print(c)'''
'''import os
f=open("ie.txt","w")
f.write("Hello World")
if os.path.exists("ie.txt"):
    os.remove("ie.txt")
    print("The file is deleted")
else:
    print("The file does not exist")
'''
with open("cpnew.txt","w") as f1,open("New.txt","r") as f2:
    c=f2.read()
    f1.write(c)