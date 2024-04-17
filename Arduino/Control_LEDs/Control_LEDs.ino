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
int STIM_STATUS = 0; //initialze a variable for if the stimulator should be active

//initialize camera variable
const int IN_STROBE = 3; //input pin to monitor when the camera exposure is active

// other variables
bool switchLight = true; // flag when expecting strobe signal. Set to false after switching LEDs and gets refreshed when strobe reads low.
bool currentBlue = true; // flag to indicate LED state. true if blue is on, false if violet is on.
bool strobeOn = false; // flag for strobe signal. useful to avoid multiple reads from trigger line.


void setup() {
  // initialize serial communication
  Serial.begin(9600);

  //initialize LED and STROBE as false
  Serial.println(VIOLET_STATUS);
  Serial.println(BLUE_STATUS);
  Serial.println(ALTERNATE_STATUS);
  Serial.println(STIM_STATUS);

  // initialize output led pins
  pinMode(OUT_VIOLET, OUTPUT);
  pinMode(OUT_BLUE, OUTPUT);
  pinMode(OUT_STIM, OUTPUT);

  digitalWrite(OUT_BLUE, LOW);
  digitalWrite(OUT_VIOLET, LOW);
  digitalWrite(OUT_STIM, LOW);

  // initialize camera input
  pinMode(IN_STROBE, INPUT);
  //strobeOn=true;
  delay(500);
}

void loop() {

  strobeOn = true; //digitalReadFast(IN_STROBE); //monitor the camera exposure
  if (!strobeOn) {
    digitalWriteFast(OUT_VIOLET,LOW);
    digitalWriteFast(OUT_BLUE,LOW);
 }

  if (STIM_STATUS==1) {
      digitalWrite(OUT_STIM, HIGH);  // use as strobe indicator
      delay(2000);
      digitalWrite(OUT_STIM, LOW);  // use as strobe indicator
      delay(4000);
  }

  if (ALTERNATE_STATUS==1) { // if in alternate mode

    if (strobeOn) { // received strobe signal
      digitalWrite(LED_BUILTIN, HIGH);  // use as strobe indicator

      if (currentBlue) {
        digitalWriteFast(OUT_BLUE,HIGH);
        digitalWriteFast(OUT_VIOLET,LOW);
      }
      else {
      digitalWriteFast(OUT_BLUE,LOW);
      digitalWriteFast(OUT_VIOLET,HIGH);
      }
      switchLight = true;
    }

    else { // camera is off so switch light if needed
      digitalWrite(LED_BUILTIN, LOW);

      // lights aren't on if the camera isn't on
      digitalWriteFast(OUT_VIOLET,LOW);
      digitalWriteFast(OUT_BLUE,LOW);

      if (switchLight) {
        if (currentBlue) {
            currentBlue = false;
          }
        else {
            currentBlue = true;
          }
        switchLight = false;
      }
    }
  }

  else if (ALTERNATE_STATUS==0) {   // directly control the LEDs programatically
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
}
