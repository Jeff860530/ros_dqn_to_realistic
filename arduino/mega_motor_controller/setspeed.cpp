#include <Arduino.h>
#include "setspeed.h"

Motor::Motor(int stp, int dir ,int rev)
{
  stp_pin = stp;
  dir_pin = dir;

  m_speed = 0;
  StepPerRev = rev;
}

void Motor::go(float s,bool d){
  digitalWrite(dir_pin,(d)?(HIGH):(LOW));

  int scape = 1000000/(s*StepPerRev);
  int Tscape = scape*2;

  temp = (micros()%Tscape >scape)?(true):(false);

  if(temp != Ttemp){
    digitalWrite(stp_pin,HIGH);
    delayMicroseconds(1);
    digitalWrite(stp_pin,LOW);
    Ttemp = temp;
  }
}

void Motor::setspeed(float set_speed){
  m_speed = 0;
}
