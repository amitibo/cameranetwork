#include <SoftwareSerial.h>
#include <Makeblock.h>
#include <Wire.h>

//Parts required:Me RJ25 Adapter and two servo   
//Me RJ25 Adapter SLOT1 connect servo1, SLOT2connect servo2,
//The Me RJ25 Adapter module can connect to the port with yellow tag (PORT_3 to PROT_8). 

#include <Servo.h> //include the Servo library;

//
// Define the pins of the servos
//
MePort port3(PORT_3);

Servo myservo1;  // create servo object to control a servo 
Servo myservo2;  // create servo object to control another servo
int servo1pin =  port3.pin1();//attaches the servo on PORT_3 SLOT1 to the servo object
int servo2pin =  port3.pin2();//attaches the servo on PORT_3 SLOT2 to the servo object

const int BUFFER_SIZE = 100;
const char* READY_REPLY = "READY";

int MAX_X_POSITION = 1000;
int MAX_Y_POSITION = 1000;

bool camera_moving = false;


void initServos()
{
    //
    // Setup the servos
    //
    if (myservo1.attached()) {
        myservo1.detach();
    }
    if (myservo2.attached()) {
        myservo2.detach();
    }
    myservo1.attach(servo1pin);  // attaches the servo on servopin1
    myservo2.attach(servo2pin);  // attaches the servo on servopin2
    
    myservo1.write(90);                  // sets the servo position according to the scaled value 
    myservo2.write(90);
}


void setup()
{
    //
    // Initialization
    //
    initServos();
    Serial.begin(9600);
    Serial.println(READY_REPLY);
}


int readCommand(char *buffer, int buffer_size)
{
    char c=0;
    int read_size;

    read_size = Serial.readBytesUntil('\n', buffer, buffer_size);

    //
    // Echo
    //
    buffer[read_size] = '\n';
    buffer[read_size+1] = '\0';
    Serial.print(buffer);

    return read_size;
}


void resetPostion(void)
{
    initServos();
    Serial.println(READY_REPLY);
}


int charToInt(char *buffer, int int_size)
{
    int i;
    int digit;
    int val = 0;

    for(i=0; i<int_size; i++)
    {
        digit = buffer[i] - int('0');
        if ((digit<0) || (digit > 9))
        {
            val = 0;
            return val;
        }
        val *= 10;
        val += digit;
    }

    return val;
}

void movePosition(char *buffer, int buffer_size)
{
    int dest1 = 0;
    int dest2 = 0;

    if (buffer_size != 9)
    {
        Serial.println("movePosition: wrong buffer_size");
        return;
    }

    dest1 = charToInt(&buffer[1], 4);
    dest2 = charToInt(&buffer[5], 4);

    myservo1.write(dest1);
    myservo2.write(dest2);
}


void readPosition(void)
{
    Serial.print("read position");
}


void loop()
{
    char Buffer[BUFFER_SIZE];
    int ByteCount;
    byte rxBuf;
    
    //
    // add main program code here
    //
    while(1)
    {
        if(Serial.available())
        {
            ByteCount = readCommand(Buffer, BUFFER_SIZE);
            if (ByteCount <= 0)
            continue;

            switch(Buffer[0])
            {
                case 'z':
                {
                    //
                    // Reset position
                    //
                    resetPostion();
                    break;
                };
                case 'm':
                {
                    //
                    // Move to position
                    //
                    movePosition(Buffer, ByteCount);
                    break;
                };
                case 'r':
                {
                    //
                    // Move to position
                    //
                    readPosition();
                    break;
                };
            }
        }
    }
}
