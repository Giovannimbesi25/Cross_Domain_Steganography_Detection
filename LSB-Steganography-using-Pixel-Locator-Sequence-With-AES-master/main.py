import LSB as lsb
import AES as Cipher
import os


def main():
    select = input("Enter E for Encoding D for Decoding :")
    if select == 'E':
        coverpath="/home/rforte/MultimediaProject/MultimediaProject/Cover/"
        outpath="/home/rforte/MultimediaProject/MultimediaProject/LSB/"
        images=os.listdir(coverpath)
        print(images)
        for i in range(0,5000):
            if os.path.exists("out.txt"):
                os.remove("out.txt")
            if os.path.exists("pls.txt.enc"):
                os.remove("pls.txt.enc")
            if os.path.exists("pls.txt"):
                os.remove("pls.txt")
            if os.path.exists("images/out1.png"):
                os.remove("images/out1.png")

            if os.path.exists("images/in1.png"):
                secretMessage = "input(Enter the secret message :)"
                passwordText = "input(Password :)"
                encodedMessage = Cipher.encrypt(secretMessage, passwordText)
                print(encodedMessage)
                lsb.LsbEncoding(coverpath+images[i],encodedMessage,outpath+images[i])
                if os.path.exists("pls.txt"):
                    os.remove("pls.txt")
            else : print("Image is not Present")



    if select == 'D':
        if os.path.exists("pls.txt.enc"):
            decodedText = lsb.LsbDecoding()
            print(decodedText)
            password = input("Enter the password :")
            finalMessage = Cipher.decrypt(decodedText, password)
            print("Final message :", finalMessage)
        else :
            print("PLS file is not present !")








main()
