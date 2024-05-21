/*
Firmware running on Teensy 4.2 used to control the LEDs and tactile stimulator

The microcontroller receives serial commands from the PC to control the LEDs and tactile stimulator.
LEDs can be turned on/off with serial commands (e.g. "SET BLUE_STATUS 1; "SET BLUE_STATUS 0;").
Or, LEDs can be set to alternate with the camera trigger signal using "SET ALTERNATE_STATUS 1;".

The tactile stimulator can be turned on/off with serial commands (e.g. "SET STIM_STATUS 1; "SET STIM_STATUS 0;").

*/


// initialize LED variables
const int OUT_VIOLET = 5; // output pin for controlling the violet LED
const int OUT_BLUE = 6; // output pin for controlling the blue LED
const int OUT_STIM = 10; // output pin for controlling the stimulator

const int MAX_CHRS = 30; //maximum number of characters to use during serial comms.
char commandBuffer[MAX_CHRS]; //initialize the command buffer

int BLUE_STATUS = 0; //initialze a variable for the blue light status
int VIOLET_STATUS = 0; //initialze a variable for the violet light status
int ALTERNATE_STATUS = 0; //initialze a variable for if the LEDs should alternate
int ALTERNATE_STATUS_OLD = 0; //used to track if the variable has changed
int STIM_STATUS = 0; //initialze a variable for if the stimulator should be active


volatile long frame_count = 0; // counter for camera sync pulses

const byte IN_STROBE = 3; //input pin to monitor when the camera exposure is active

// switch LEDs when camera trigger is detected (for alternating mode)
void camera_triggered() {
    frame_count++;
    if (frame_count%2 == 0){
      digitalWriteFast(OUT_VIOLET,LOW);
      digitalWriteFast(OUT_BLUE,HIGH);
    }
    else{
      digitalWriteFast(OUT_VIOLET,HIGH);
      digitalWriteFast(OUT_BLUE,LOW);          
    }


}

void setup() {
  // initialize serial communication
  Serial.begin(57600);

  //initialize LED and STROBE as false
  Serial.println(VIOLET_STATUS);
  Serial.println(BLUE_STATUS);
  Serial.println(ALTERNATE_STATUS);

  
  // initialize output led pins
  pinMode(OUT_VIOLET, OUTPUT);
  pinMode(OUT_BLUE, OUTPUT);
  pinMode(OUT_STIM, OUTPUT);

  // initialize camera input
  pinMode(IN_STROBE, INPUT_PULLUP);

}

// Serial handler
void serialEvent() {
  // concatinate incoming characters from the serial port
  while (Serial.available() > 0) {
    char c = Serial.read();
    // Add any characters that aren't the end of a command (semicolon) to the input buffer.
    if (c != ';') {
      c = toupper(c);
      strncat(commandBuffer, &c, 1);
    }
    else
    {
      // Parse the command because an end of command token was encountered.
      Serial.println(commandBuffer);
      parseCommand(commandBuffer);


      // Clear the input buffer
      memset(commandBuffer, 0, sizeof(commandBuffer));
    }
  }
}

// Either get variable name or asynchronously set variable
#define GET_AND_SET(variableName) \
  if (strstr(command, "GET " #variableName) != NULL) { \
    Serial.print(#variableName" "); \
    Serial.println(variableName); \
  } \
  else if (strstr(command, "SET " #variableName " ") != NULL) { \
    variableName = (typeof(variableName)) atof(command+(sizeof("SET " #variableName " ")-1)); \
    Serial.print(#variableName" "); \
    Serial.println(variableName); \
  }

// react to serial command by getting or setting the variables
void parseCommand(char* command) {

  GET_AND_SET(BLUE_STATUS);
  GET_AND_SET(VIOLET_STATUS);
  GET_AND_SET(ALTERNATE_STATUS);
  GET_AND_SET(STIM_STATUS);
}


void loop() {
  
  // if alternate status, attach inteerupt to camera trigger to switch LEDs
  if ((ALTERNATE_STATUS == 1) & (ALTERNATE_STATUS_OLD == 0)){ // first instance of alternate mode
    attachInterrupt(digitalPinToInterrupt(IN_STROBE), camera_triggered, RISING);
    ALTERNATE_STATUS_OLD = 1;
  }

  // directly control the LEDs with serial commands
  if (ALTERNATE_STATUS==0) {   
    if (BLUE_STATUS==1) {
      digitalWriteFast(OUT_BLUE, HIGH);
    }
    if (BLUE_STATUS==0) {
      digitalWriteFast(OUT_BLUE, LOW);
    }
    if (VIOLET_STATUS==1) {
      digitalWriteFast(OUT_VIOLET, HIGH);
    }
    if (VIOLET_STATUS==0) {
      digitalWriteFast(OUT_VIOLET, LOW);
    }
  }

  // tactile stimulator control
  if (STIM_STATUS==1) {
      digitalWriteFast(OUT_STIM, HIGH);
  }
  else {
      digitalWriteFast(OUT_STIM, LOW);
  }
}