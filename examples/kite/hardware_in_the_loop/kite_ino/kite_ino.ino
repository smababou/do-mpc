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

	// initialize control
	ctl.in[0] = 0.34359907;
	ctl.in[1] = 0.52791537;
	ctl.in[2] = 0.0;

	// initilaze ekf
	ctl.ekf->x_hat[0] = 0.34359907;
	ctl.ekf->x_hat[1] = 0.52791537;
	ctl.ekf->x_hat[2] = 0.0;
	ctl.ekf->x_hat[3] = 5.0;
	ctl.ekf->x_hat[4] = 10.0;

}

void loop() {

	char theta[100];
	char phi[100];
	uint32_t i;

	// make optimization step
	make_dnn_step(&ctl);
	ctl.out[0] = fmin(fmax(ctl.out[0],-10.0),10.0);

	// make projection step
	make_projection_step(&ctl);
 	ctl.out[0] = fmin(fmax(ctl.out[0],-10.0),10.0);

	// return optimal control input
	Serial.println(ctl.out[0],8);

	// estimation loop
	for (i=0; i<3; i++) {

		// wait for new information
		while (Serial.available() == 0) {
			delay(5);
		}

		// get data from simulation (measurements)
		while (Serial.available() > 0) {
			Serial.readBytes(theta,100);
		}

		while (Serial.available() == 0) {
			delay(5);
		}

		while (Serial.available() > 0) {
			Serial.readBytes(phi,100);
		}

		// convert char arrays to float and pass to ekf as measurement
		ctl.ekf->y[0] = (real_t) atof(theta);
		ctl.ekf->y[1] = (real_t) atof(phi);

		// make estimation step
		make_ekf_step(&ctl);

		Serial.println("continue");

	}

	// return estimated states and parameters and the optimal input
	Serial.println(ctl.ekf->x_hat[0],8);
	Serial.println(ctl.ekf->x_hat[1],8);
	Serial.println(ctl.ekf->x_hat[2],8);
	Serial.println(ctl.ekf->x_hat[3],8);
	Serial.println(ctl.ekf->x_hat[4],8);

	// pass result of ekf to optimizer (NN)
	ctl.in[0] = ctl.ekf->x_hat[0];
	ctl.in[1] = ctl.ekf->x_hat[1];
	ctl.in[2] = ctl.ekf->x_hat[2];

}
