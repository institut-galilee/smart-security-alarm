/*
*
* Arduino Security and Alarm System (With FaceReconize)
* By Nabil AMEZIANE & Abderrahmane MRABET
*
*/

#include <LiquidCrystal.h> // includes the LiquidCrystal Library 
#include <Keypad.h> // includes  the  Keypad Library
 
#define buzzer 8
#define trigPin 9
#define echoPin 10
#define ledRed 11
#define ledGreen 12

int val = 0;
int sensorPin = 0;
int incomingByte;
 
long duration;
int distance, initialDistance, currentDistance, i;
int screenOffMsg =0;
String password="1234";
String tempPassword;
boolean activated = false; // State of the alarm
boolean isActivated;
boolean activateAlarm = false;
boolean alarmActivated = false;
boolean enteredPassword; // State of the entered password to stop the alarm
boolean passChangeMode = false;
boolean passChanged = false;

const byte ROWS = 4; //four rows
const byte COLS = 4; //four columns
char keypressed;
//define the cymbols on the buttons of the keypads
char keyMap[ROWS][COLS] = {
  {'1','2','3','A'},
  {'4','5','6','B'},
  {'7','8','9','C'},
  {'*','0','#','D'}
};

byte rowPins[ROWS] = {14, 15, 16, 17}; //Row pinouts of the keypad
byte colPins[COLS] = {18, 19, 20, 21}; //Column pinouts of the keypad

Keypad myKeypad = Keypad( makeKeymap(keyMap), rowPins, colPins, ROWS, COLS); 
LiquidCrystal lcd(2, 3, 4, 5, 6, 7); // Creates an LC object. Parameters: (rs, enable, d4, d5, d6, d7) 



#include "pitches.h"

// notes in the melody:
int melody[] = {
  NOTE_C4, NOTE_G3, NOTE_G3, NOTE_A3, NOTE_G3, 0, NOTE_B3, NOTE_C4
};

// note durations: 4 = quarter note, 8 = eighth note, etc.:
int noteDurations[] = {
  4, 8, 8, 4, 4, 4, 4, 4
};


void setup() { 

  Serial.begin(9600);
  
  lcd.begin(16,2);
  lcd.setCursor(0,0);
  lcd.print("* Alarm Security *");
  lcd.setCursor(0,1);
  lcd.print("(Face-Reconize)");
  digitalWrite(ledRed, HIGH);
  digitalWrite(ledGreen, LOW);
  delay(2000);
  
  lcd.clear(); 
  lcd.setCursor(0,0);
  lcd.print("Designed By :");
  lcd.setCursor(0,1);
  lcd.print("****************");
  digitalWrite(ledRed, LOW);
  digitalWrite(ledGreen, HIGH);
  delay(2000);

  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("=> N.AMEZIANE");
  lcd.setCursor(0,1);
  lcd.print("=> A.MBRABET");
  digitalWrite(ledRed, HIGH);
  digitalWrite(ledGreen, LOW);
  delay(2000);



  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("*** WELCOME ***");
  digitalWrite(ledRed,LOW );
  digitalWrite(ledGreen,HIGH );
  delay(2000);  
  digitalWrite(ledGreen,LOW );
  
  song();
  
  pinMode(buzzer, OUTPUT); // Set buzzer as an output
  pinMode(trigPin, OUTPUT); // Sets the trigPin as an Output
  pinMode(echoPin, INPUT); // Sets the echoPin as an Input
}


void loop() {
  if (activateAlarm) {
    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("Alarm will be");
    lcd.setCursor(0,1); 
    lcd.print("activated in");
   
    int countdown = 3; // 3 seconds count down before activating the alarm
    while (countdown != 0) {
      lcd.setCursor(13,1);
      lcd.print(countdown);
      countdown--;
      tone(buzzer, 700, 100);
      delay(1000);
    }
    
    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("Alarm Activated!");
    digitalWrite(ledGreen, HIGH);
    initialDistance = getDistance();
    activateAlarm = false;
    alarmActivated = true;
  }

  if (alarmActivated == true){
      currentDistance = getDistance() + 10;
      if ( currentDistance < initialDistance) {
        tone(buzzer, 1000); // Send 1KHz sound signal 
        digitalWrite(ledRed, HIGH);
        digitalWrite(ledGreen, LOW);
        val = analogRead(sensorPin);
        Serial.println(val);
        incomingByte = Serial.read();
        enterPassword();
      }
  }

  if (!alarmActivated) {
    if (screenOffMsg == 0 ){
      lcd.clear();
      lcd.setCursor(0,0);
      lcd.print("A - Activate");
      lcd.setCursor(0,1);
      lcd.print("B - Change Pass");
      screenOffMsg = 1;
    }
    keypressed = myKeypad.getKey();
     if (keypressed =='A'){        //If A is pressed, activate the alarm
      tone(buzzer, 1000, 200);
      activateAlarm = true;            
    }
    else if (keypressed =='B') {
      lcd.clear();
      int i=1;
      tone(buzzer, 2000, 100);
      tempPassword = "";
      lcd.setCursor(0,0);
      lcd.print("Current Password");
      lcd.setCursor(0,1);
      lcd.print(">");
      passChangeMode = true;
      passChanged = true;   
      while(passChanged) {      
      keypressed = myKeypad.getKey();
      if (keypressed != NO_KEY){
        if (keypressed == '0' || keypressed == '1' || keypressed == '2' || keypressed == '3' ||
            keypressed == '4' || keypressed == '5' || keypressed == '6' || keypressed == '7' ||
            keypressed == '8' || keypressed == '9' ) {
         tempPassword += keypressed;
         lcd.setCursor(i,1);
         lcd.print("*");
         i++;
         tone(buzzer, 2000, 100);
        }
      }
      if (i > 5 || keypressed == '#') {
        tempPassword = "";
        i=1;
        lcd.clear();
        lcd.setCursor(0,0);
        lcd.print("Current Password");
        lcd.setCursor(0,1);
        lcd.print(">"); 
      }
      if ( keypressed == '*') {
        i=1;
        tone(buzzer, 2000, 100);
        if (password == tempPassword) {
          tempPassword="";
          lcd.clear();
          lcd.setCursor(0,0);
          lcd.print("Set New Password");
          lcd.setCursor(0,1);
          lcd.print(">");
          while(passChangeMode) {
            keypressed = myKeypad.getKey();
            if (keypressed != NO_KEY){
              if (keypressed == '0' || keypressed == '1' || keypressed == '2' || keypressed == '3' ||
                  keypressed == '4' || keypressed == '5' || keypressed == '6' || keypressed == '7' ||
                  keypressed == '8' || keypressed == '9' ) {
                tempPassword += keypressed;
                lcd.setCursor(i,1);
                lcd.print("*");
                i++;
                tone(buzzer, 2000, 100);
              }
            }
            if (i > 5 || keypressed == '#') {
              tempPassword = "";
              i=1;
              tone(buzzer, 2000, 100);
              lcd.clear();
              lcd.setCursor(0,0);
              lcd.print("Set New Password");
              lcd.setCursor(0,1);
              lcd.print(">");
            }
            if ( keypressed == '*') {
              i=1;
              tone(buzzer, 2000, 100);
              password = tempPassword;
              passChangeMode = false;
              passChanged = false;
              screenOffMsg = 0;
            }            
          }
        }
      }
    }
   }
 }
}


void enterPassword() {
  int k=5;
  tempPassword = "";
  activated = true;
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print(" *** ALARM *** ");
  lcd.setCursor(0,1);
  lcd.print("Pass>");
      while(activated) {
      keypressed = myKeypad.getKey();
      if (keypressed != NO_KEY){
        if (keypressed == '0' || keypressed == '1' || keypressed == '2' || keypressed == '3' ||
            keypressed == '4' || keypressed == '5' || keypressed == '6' || keypressed == '7' ||
            keypressed == '8' || keypressed == '9' ) {
          tempPassword += keypressed;
          lcd.setCursor(k,1);
          lcd.print("*");
          k++;
        }
      
      }else if (Serial.available() > 0) {
 	if(incomingByte = 1){
                digitalWrite(ledRed, LOW);
                digitalWrite(ledGreen, HIGH);
                lcd.clear();
                lcd.setCursor(0,0);
                lcd.print("* FACE DETECTED *");
                lcd.setCursor(0,1);
                lcd.print("ALARM DESACTIVATED");
                delay(1000);
                digitalWrite(ledGreen, LOW);
                noTone(buzzer);
                song();
                activated = false;
                alarmActivated = false;
                screenOffMsg = 0;
      	}
          
      }
      
      
      if (k > 9 || keypressed == '#') {
        tempPassword = "";
        k=5;
        lcd.clear();
        lcd.setCursor(0,0);
        lcd.print(" *** ALARM *** ");
        lcd.setCursor(0,1);
        lcd.print("Pass>");
      }
      if ( keypressed == '*') {
        if ( tempPassword == password ) {
          digitalWrite(ledRed, LOW);
          digitalWrite(ledGreen, HIGH);
          lcd.clear();
          lcd.setCursor(0,0);
          lcd.print("* PASSWORD OK * ");
          lcd.setCursor(0,1);
          lcd.print("ALARM DESACTIVATED");
          delay(1000);
          digitalWrite(ledGreen, LOW);
          activated = false;
          alarmActivated = false;
          noTone(buzzer);
          screenOffMsg = 0; 
        }
        else if (tempPassword != password) {
          lcd.setCursor(0,1);
          lcd.print("Wrong! Try Again");
          delay(2000);
          lcd.clear();
          lcd.setCursor(0,0);
          lcd.print(" *** ALARM *** ");
          lcd.setCursor(0,1);
          lcd.print("Pass>");
        }
      }    
    }
}


// Custom function for the Ultrasonic sensor
long getDistance(){
  //int i=10;
  
  //while( i<=10 ) {
  // Clears the trigPin
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // Sets the trigPin on HIGH state for 10 micro seconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin, HIGH);

  // Calculating the distance
  distance = duration*0.034/2;
  //sumDistance += distance;
  //}
  //int averageDistance= sumDistance/10;
  return distance;

}


void song(){
    for (int thisNote = 0; thisNote < 8; thisNote++) {

    // to calculate the note duration, take one second divided by the note type.
    //e.g. quarter note = 1000 / 4, eighth note = 1000/8, etc.
    int noteDuration = 1000 / noteDurations[thisNote];
    tone(8, melody[thisNote], noteDuration);

    // to distinguish the notes, set a minimum time between them.
    // the note's duration + 30% seems to work well:
    int pauseBetweenNotes = noteDuration * 1.30;
    delay(pauseBetweenNotes);
    // stop the tone playing:
    noTone(8);
    }
}



