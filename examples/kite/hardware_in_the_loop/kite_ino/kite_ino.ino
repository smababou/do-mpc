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

  // initilaze ekf
  ctl.ekf->x_hat[0] = 0.34359907;
  ctl.ekf->x_hat[1] = 0.52791537;
  ctl.ekf->x_hat[2] = 0.0;
	real_t u_mpc = 1.22835618;
	ctl.out[0] = u_mpc;
}

void loop() {

	int index = 0;
	char theta[100];
	char phi[100];
//	char psi[100];

	// wait for new information
	while (Serial.available() == 0) {
		delay(5);
	}

	// get data from simulation (measurements)

	while (Serial.available() > 0) {
		Serial.readBytes(theta,100);
	}

	index = 0;
	while (Serial.available() == 0) {
		delay(5);
	}

	while (Serial.available() > 0) {
		Serial.readBytes(phi,100);
	}

//	index = 0;
//	while (Serial.available() == 0) {
//		delay(5);
//	}
//
//	while (Serial.available() > 0) {
//		Serial.readBytes(psi,100);
//	}

	ctl.ekf->y[0] = (float32_t) atof(theta);
	ctl.ekf->y[1] = (float32_t) atof(phi);
//	ctl.in[2] = (float32_t) atof(psi);

	// make estimation step
	make_ekf_step(&ctl);

	// make optimization step
	make_dnn_step(&ctl);

	// return optimal input (and for plotting the estimated values)
	float print_out = (float) ctl.out[0];
	Serial.print(print_out,4);

}
