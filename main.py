from FR.face_recognition import recognize
from Object_Detection.object_detection import detection

def main():
    ask = int(input("What do you want to test?\n1. Face recognition\n2. Object detection\n3. Quit\nYour command: "))
    if ask== 1:
        reg = recognize()
        train =  int(input("What do you want to re-train models?\n1. Yes\n2. No\nYour command: "))
        if train == 1:
            reg.train()
        elif train == 2:
            print("No training!")
        else:
            print('Wrong input')
            main()
        reg.live_testing()
    elif ask == 2:
        detect = detection()
        command = int(input("What test do you want to run?\n1. Image\n2. Live testing\nYour command: "))
        if command == 1:
            detect.get_input()
        elif command == 2:
            detect.live_testing()
        else:
            print('Wrong input')
            main()
    elif ask == 3:
        pass
    else:
        print("Wrong input")
        main()

if __name__ == "__main__":
    main()