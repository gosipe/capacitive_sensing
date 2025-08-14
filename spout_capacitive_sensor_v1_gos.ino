#include <CapacitiveSensor.h>
CapacitiveSensor Spout = CapacitiveSensor(2,4);
void setup()                    
{
   //Spout.set_CS_AutocaL_Millis(0xFFFFFFFF);     // turn off autocalibrate on channel 1 - just as an example
   Serial.begin(9600);
}

void loop()                    
{
    long start = millis();
    long total1 =  Spout.capacitiveSensor(10);

    Serial.print(millis() - start);        // check on performance in milliseconds
    Serial.print("\t");                    // tab character for debug windown spacing

    Serial.println(total1);                  // print sensor output 1
    

    delay(50);                             // arbitrary delay to limit data to serial port 
}
