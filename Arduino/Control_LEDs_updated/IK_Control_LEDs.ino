// Michael Reitman December 20th 2023
// Teensy code to control the LEDs for a widefield microscope
// adapted from Dave Mets's arcadia-phenotypomat and Joao Coutou's labcams


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


volatile long pulse_count = 0; // number of pulses to trigger
volatile long frame_count = 0; // counter for camera sync pulses

//initialize camera variable
const byte IN_STROBE = 3; //input pin to monitor when the camera exposure is active

// other variables
bool noSwitch = false; // flag for dead-time
bool switchLight = true; // flag when expecting strobe signal. Set to false after switching LEDs and gets refreshed when strobe reads low.
bool currentBlue = true; // flag to indicate LED state. true if blue is on, false if violet is on.
unsigned long clocker = millis(); // timer to create dead-time after receiving a strobe signal.
bool strobeOn = false; // flag for strobe signal. useful to avoid multiple reads from trigger line.


void camera_triggered() {
  //if (digitalReadFast(IN_STROBE) == LOW) {

  //}
  //else {
    frame_count++;
    if (frame_count%2 == 0){
      digitalWriteFast(OUT_VIOLET,LOW);
      digitalWriteFast(OUT_BLUE,HIGH);
    }
    else{
      digitalWriteFast(OUT_VIOLET,HIGH);
      digitalWriteFast(OUT_BLUE,LOW);          
    }
    
  //}

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

void loop() {
  
  // if alternate status, attach inteerupt to camera trigger to switch LEDs
  if ((ALTERNATE_STATUS == 1) & (ALTERNATE_STATUS_OLD == 0)){ // first instance of alternate mode
    attachInterrupt(digitalPinToInterrupt(IN_STROBE), camera_triggered, RISING);
    ALTERNATE_STATUS_OLD = 1;
  }

  if (ALTERNATE_STATUS==0) {   // directly control the LEDs programatically
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

  // deal with stim
  if (STIM_STATUS==1) {
      digitalWriteFast(OUT_STIM, HIGH);
  }
  else {
      digitalWriteFast(OUT_STIM, LOW);
  }
}