lerobot-calibrate \
  --robot.type=so101_follower \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C123878-if00 \  bullsht
  --robot.id=my_awesome_follower_arm

lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C123878-if00 \
  --teleop.id=my_awesome_leader_arm1


lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128167-if00 \
  --robot.id=my_awesome_follower_arm1

find ~/.cache/huggingface/lerobot/calibration/robots ~/.cache/huggingface/lerobot/calibration/teleoperators -type f -delete





lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm

  78 is leader

lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C123878-if00 \
  --teleop.id=my_awesome_leader_arm


lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128167-if00 \
  --robot.id=my_awesome_follower_arm

  5C4C123878

lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 # <- The port of your robot
  --robot.id=my_awesome_follower_arm  # <- Give the robot a unique name

  lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 # <- The port of your robot
  --teleop.id=my_awesome_leader_arm  # <- Give the robot a unique name



  lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm


lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm \
  --display_data=true

conda activate lerobot
sudo chmod 666 /dev/ttyACM*
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm

 
 