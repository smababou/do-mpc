extern "C" {
#include "edgeAI_main.h"
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern struct edgeAI_ctl ctl;

void setup() {
	Serial.begin(9600);
	while (!Serial);
  Serial.setTimeout(20);

//	const float32_t IN0 = 0.34359907, IN1 = 0.52791537, IN2 = 0.0;
//	ctl.in[0] = IN0;
//	ctl.in[1] = IN1;
//	ctl.in[2] = IN2;

}
	
void loop() {
  
  int index = 0;
  char theta[100];
  char phi[100];
  char psi[100];

//  if (Serial.available() > 0) {
//    delay(20);
//  }

  // wait for new information
  while (Serial.available() == 0) {
    delay(5);
  }
  
  // get data from simulation (measurements)
  
  while (Serial.available() > 0) {
//      theta[index++] = Serial.read();
      Serial.readBytes(theta,100);
  }
  
  index = 0;
  while (Serial.available() == 0) {
    delay(5);
  }

//  if (Serial.available() > 0) {
//    delay(20);
//  }
  
  while (Serial.available() > 0) {
//      phi[index++] = Serial.read();
    Serial.readBytes(phi,100);
  }
  
  index = 0;
  while (Serial.available() == 0) {
    delay(5);
  }

//  if (Serial.available() > 0) {
//    delay(20);
//  }
  
  while (Serial.available() > 0) {
//      psi[index++] = Serial.read();
      Serial.readBytes(psi,100);
  }
  
  ctl.in[0] = (float32_t) atof(theta);
  ctl.in[1] = (float32_t) atof(phi);
  ctl.in[2] = (float32_t) atof(psi);

  // make optimization step
	make_dnn_step(&ctl);

  // return optimal input
  float print_out = (float) ctl.out[0];
	Serial.print(print_out,4);

}
