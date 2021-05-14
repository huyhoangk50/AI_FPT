pip3 install gdown
mkdir model && cd model &&\
mkdir unet && cd unet &&\
# model unet
gdown --id 1QXEJbIKOoKdHeffpZuPoFndK14OdH9zf && cd .. && \
mkdir yolo_model && cd yolo_model &&\
# obj.data
gdown --id 1imcijco7Yqmi3fFWANdFKkN5y5-UoQem && \
# obj.names
gdown --id 1yx66kN6A3gcczpRIvOxXw4HAN7vLU0vw && \
# config
gdown --id 1tfeCEqL8h4_ZPwlPd1x7YRDVg8cjIled && \
# weights 
gdown --id 1819g96vPr0Ws79lhWNPC1132RqiCeG6d