#!/usr/bin/env python
import glob

import rospy

import message_filters

"""
A ROS node to identify known users using face_recognition Python library.

This is a FILTER, taking in body_tracker_msgs/BodyTrackerArray messages, and 
outputting the same message, with names of any identified people added

The people who are placed in /people directory will be automatically fetched
and their face features will be compared to incoming face images. If a similar
face is found, the name of the closest face image will be assigned to that
bounding box.

This is modified code from example by:        
    Cagatay Odabasi, ros_people_object_detection_tensorflow 2017. GitHub repository: 
    https://github.com/cagbal/ros_people_object_detection_tensorflow
    cagatay.odabasi@ipa.fraunhofer.de    
"""

import numpy as np

import cv2

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

#from cob_perception_msgs.msg import DetectionArray
from body_tracker_msgs.msg import BodyTrackerArray

import face_recognition as fr

import rospkg
import logging


# Get the package directory
rospack = rospkg.RosPack()
cd = rospack.get_path('face_recognizer')

cv2.namedWindow('Face Recognition')

class FaceRecognitionNode(object):
    """A ROS node to get face bounding boxes inside of person bounding boxes

    _bridge (CvBridge): Bridge between ROS and CV image
    pub_det (Publisher): Publisher object for detections (bounding box, name labels)
    pub_det_rgb (Publisher): Publisher object for detection image
    sub_detection (Subscriber): Subscriber object for object_detection
    sub_image (Subscriber): Subscriber object for RGB image from camera
    scaling_factor (Float): Input image will be scaled down with this
    database (List): Contains face features of people inside /people folder

    """
    def __init__(self):
        super(FaceRecognitionNode, self).__init__()

        # init the node
        rospy.init_node('face_recognition_node', anonymous=False)
        rospy.loginfo("DBG Starting face_recognition_node...")
        # Get the parameters
        (image_topic, detection_topic, output_topic, output_topic_rgb) \
            = self.get_parameters()

        rospy.loginfo("Listening on Detection Topic: " + detection_topic)
        rospy.loginfo("and listening on Image Topic: " + image_topic)

        self._bridge = CvBridge()

        # Advertise the result of Object Tracker
        self.pub_det = rospy.Publisher(output_topic, \
            BodyTrackerArray, queue_size=1)

        self.pub_det_rgb = rospy.Publisher(output_topic_rgb, \
            Image, queue_size=1)


        self.sub_detection = message_filters.Subscriber(detection_topic, \
            BodyTrackerArray)
        self.sub_image = message_filters.Subscriber(image_topic, Image)

        # Scaling factor for face recognition image
        self.scaling_factor = 1.0 # 0.50

        # Read the images from folder and create a database
        self.database = self.initialize_database()

        ts = message_filters.ApproximateTimeSynchronizer(\
            [self.sub_detection, self.sub_image], 2, 0.2)

        ts.registerCallback(self.detection_callback)

        # spin
        rospy.spin()
             

    def get_parameters(self):
        """
        Gets the necessary parameters from parameter server

        Args:

        Returns:
        (tuple) (camera_topic, detection_topic, output_topic)

        """

        self.show_faces_at_launch = rospy.get_param("~show_faces_at_launch")
        camera_topic = rospy.get_param("~camera_topic")
        detection_topic = rospy.get_param("~detection_topic")
        output_topic = rospy.get_param("~output_topic")
        output_topic_rgb = rospy.get_param("~output_topic_rgb")

        return (camera_topic, detection_topic, output_topic, output_topic_rgb)


    def shutdown(self):
        """
        Shuts down the node
        """
        rospy.signal_shutdown("See ya!")

    def detection_callback(self, detections, image):
        """
        Callback for RGB images and detections

        Args:
        detections (body_tracker_msgs/BodyTrackerArray) : detections array --> detected_list
        image (sensor_msgs/Image): RGB image from camera

        """

        #rospy.loginfo("DBG detection_callback()")

        try:
            cv_rgb = self._bridge.imgmsg_to_cv2(image, "passthrough")[:, :, ::-1]
        except CvBridgeError as e:
            print(e)        

        # cv_rgb = cv2.resize(cv_rgb, (0, 0), fx=self.scaling_factor, fy=self.scaling_factor)

        cv_rgb=cv_rgb.astype(np.uint8)

        # Find Faces, see if we recognize them
        (cv_rgb, detections) = self.recognize(detections, cv_rgb)

        image_outgoing = self._bridge.cv2_to_imgmsg(cv_rgb, encoding="passthrough")

        self.publish(detections, image_outgoing)


    def recognize(self, msg_in, image):
        """
        Main face recognition logic, it gets the incoming detection message and
        modifies the person labeled detections according to the face info.
        """
        #rospy.loginfo("DBG recognize()")

        for i, person in enumerate(msg_in.detected_list):

            bb_left = bb_top = bb_right =  bb_bottom = 0
            
            if person.face_found:
                #rospy.loginfo("DBG face found")

                bb_left =   int(  person.face_left * self.scaling_factor)
                bb_top =    int(  person.face_top  * self.scaling_factor)
                bb_right =  int( (person.face_left + person.face_width) \
                    * self.scaling_factor)
                bb_bottom = int( (person.face_top + person.face_height)  \
                    * self.scaling_factor)

                # Crop image to face
                temp_image = image[bb_top:bb_bottom, bb_left:bb_right ]

                # Show crop person image - BGR GREEN
                cv2.rectangle(image, (bb_left, bb_top), (bb_right, bb_bottom), (0, 255, 0), 3)

                # DEBUG Show frame size on image - BGR WHITE
                # format: (left, top), (right, bottom)
                cv2.rectangle(image, (30, 10), (838, 470), (255, 255, 255), 2)

                try:

                    face_locations = fr.face_locations(temp_image)

                    face_features = fr.face_encodings(temp_image, \
                        face_locations)

                    #rospy.loginfo("DBG looking for faces")

                    for features, (top, right, bottom, left) in \
                        zip(face_features, face_locations):
                        matches = fr.compare_faces(self.database[0], features)

                        rospy.loginfo("DBG found a face, looking for matching faces")
                        person.name = ""
                        name_label = "Unknown"

                        if True in matches:
                            ind = matches.index(True)
                            person.name = self.database[1][ind] # modify the message!
                            name_label = self.database[1][ind] # modify the label for display
                            rospy.loginfo("********** DBG FOUND MATCHING FACE! ********")
                            rospy.loginfo("Name = " + name_label)

                        # Draw bounding boxes on current image
                        l = bb_left + left # map into the face rectangle
                        t = bb_top + top
                        r = bb_left + right
                        b = bb_top + bottom

                        cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 2) # BGR Red

                        #cv2.rectangle(image, (x, y), \
                        #(x + width, y + height), (255, 0, 0), 3)

                        cv2.putText(image, name_label, \
                        (l + 2, t + 2), \
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)

                        #detections_out.detected_list.append(detection)

                except Exception as e:
                    print e
            #image = temp_image.copy() # debug
        
        return (image, msg_in)  # Modified message

    def publish(self, detections, image_outgoing):
        """
        Creates the ros messages and publishes them

        Args:
        detections (cob_perception_msgs/DetectionArray): incoming detections
        image_outgoing (sensor_msgs/Image): with face bounding boxes and names

        """

        self.pub_det.publish(detections)
        self.pub_det_rgb.publish(image_outgoing)



    def initialize_database(self):
        """
        Reads the PNG images from ./people folder and
        creates a list of peoples

        The names of the image files are considered as their
        real names.

        For example;
        /people
          - mario.png
          - jennifer.png
          - melanie.png

        Returns:
        (tuple) (people_list, name_list) (features of people, names of people)

        """
        filenames = glob.glob(cd + '/people/*.png')

        people_list = []
        name_list = []

        for f in filenames:
            im = cv2.imread(f, 1)

            if self.show_faces_at_launch:
                cv2.imshow('Face Recognition', im)
                cv2.waitKey(50)
                cv2.destroyAllWindows()

            im = im.astype(np.uint8)
            people_list.append(fr.face_encodings(im)[0])
            name_list.append(f.split('/')[-1].split('.')[0])

        return (people_list, name_list)

def main():
    """ main function
    """
    node = FaceRecognitionNode()

if __name__ == '__main__':
    main()
