import rospy
from std_msgs.msg import String


def talker():
    pub = rospy.Publisher('/command_cl/mode', String, queue_size=1)
    rospy.init_node('commands_cl', anonymous=False)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        print('enter working mode for segmentation algorithm:')
        mode = input()
        pub.publish(mode)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
