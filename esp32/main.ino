#define SERIAL_SIZE_RX  512
#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>
#define ONBOARD_LED  2

// sine_model.h contains the array you exported from Python with xxd or tinymlgen
#include "ECGModel.h"

#define N_INPUTS 1
#define N_OUTPUTS 1
// in future projects you may need to tweak this value: it's a trial and error process
#define TENSOR_ARENA_SIZE 5*1024

Eloquent::TinyML::TensorFlow::TensorFlow<N_INPUTS, N_OUTPUTS, TENSOR_ARENA_SIZE> tf;

int MAX_SIZE = 76;
int current_index;
bool stop_read = false;
float num;

//float X_test[76] = {-162.0,-247.0,-230.0,-203.0,-286.0,-222.0,-156.0,-174.0,533.0,-256.0,-199.0,-77.0,61.0,45.0,10.0,45.0,107.0,111.0,148.0,195.0,174.0,547.0,103.0,122.0,171.0,109.0,22.0,-259.0,-344.0,-618.0,-632.0,-447.0,-245.0,-60.0,-76.0,711.0,-209.0,-115.0,28.0,171.0,224.0,101.0,115.0,99.0,106.0,100.0,113.0,83.0,61.0,50.0,466.0,-93.0,-51.0,2.0,99.0,11.0,-3.0,-6.0,1.0,0.0,-4.0,3.0,35.0,46.0,-33.0,872.0,-122.0,-102.0,-62.0,80.0,7.0,-32.0,-49.0,2.0,-11.0,7.0};

float input_tensor[76];

String buf;

void setup() {
    delay(1000);
    buf = "";
    pinMode(ONBOARD_LED,OUTPUT);
    digitalWrite(ONBOARD_LED,LOW);
    Serial.setRxBufferSize(SERIAL_SIZE_RX);
    Serial.begin(115200);
    tf.begin(model_data);
    current_index = 0;    
    // check if model loaded fine
    if (!tf.isOk()) {
        Serial.print("ERROR: ");
        Serial.println(tf.getErrorMessage());        
        while (true) delay(1000);
    }
    delay(1000);
}

void loop() {
  while (Serial.available() > 0)
   {
     char inByte = (char)Serial.read();
     // 44 == ','
     if (inByte==44){
      float value = buf.toFloat();
      input_tensor[current_index] = value;
      buf = "";
      current_index++;
     }
     else {
      buf += inByte;
     }
   }
  if(current_index>=MAX_SIZE){
//    delay(500);
//    digitalWrite(ONBOARD_LED,HIGH);
//    delay(500);
//    digitalWrite(ONBOARD_LED,LOW);
    float predicted = tf.predict(input_tensor);
    Serial.print(predicted);
    current_index = 0;
  }  
}
