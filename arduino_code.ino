void setup() {
  Serial.begin(115200);
}

void loop() {
  int thumb = analogRead(15);
  int index = analogRead(2);
  int middle = analogRead(4);

  Serial.print(thumb);
  Serial.print(",");
  Serial.print(index);
  Serial.print(",");
  Serial.println(middle);

  delay(100);  // 10 readings per second
}
