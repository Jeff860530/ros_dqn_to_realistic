#ifndef setspeed_h
#define setspeed_h


class Motor{
  public:
    Motor(int stp, int dir ,int rev);
    void setspeed(float set_speed);
    void go(float s,bool d);
  private:
    int stp_pin;
    int dir_pin;
    float m_speed;
    int StepPerRev;
    bool temp = false;
    bool Ttemp = false;
};

#endif
