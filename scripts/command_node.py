import rospy
from std_msgs.msg import String
from pytimedinput import timedInput


def talker():
    pub = rospy.Publisher('/command_from_human', String, queue_size=1)
    rospy.init_node('commands_cl', anonymous=False)

    rate = rospy.Rate(10)
    mode = 'inference'
    while not rospy.is_shutdown():
        print('enter working mode for segmentation algorithm:')
        if 'train' in mode:
            userText, timedOut = timedInput(
                '', timeOut=15)
            if(timedOut):
                print('changing mode to inference')
                mode = 'inference'
            else:
                mode = userText
        else:
            mode = input()
        pub.publish(mode)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass


# class CV:

#     def __init__(self) -> None:
#         self.hri_srv = rospy.Service()

#         self.classificator = Classificator()
#         self.traner = Trainer()

#         self.state = 'INIT'

#     def hri_handler(self, req):
#         if req.int8 == 1:
#             self.state = 'C'
#         elif req.int8 == 2:
#             self.state = 'T'
#         else:
#             self.state = 'ERROR'

    # def image_callback(self, msg):
    #     self.cv_image = ....

    # def spin(self):

    #     rate = rospy.Rate(30)
    #     while not rospy.is_shutdown():

    #         if self.state == 'C':
    #             pairs_list, image, masked_image = self.classificator.do(self.cv_image)
    #             self.masked_rgbd_pub.publish(masked_image)
    #         elif self.state == 'T':
    #             self.taining.do()

    #         rate.sleep()

# HW -> API -> ros_API -> user_stuff
