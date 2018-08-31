id=$( \
docker run --runtime=nvidia -d -p 8888:8888 \
--mount source=data,target=/data \
--mount type=bind,source=$HOME/fp_img,target=/fp_img,readonly \
--mount type=bind,source=$HOME/notebooks,target=/notebooks \
fp_analysis \
)
sleep 10s
docker exec $id jupyter notebook list
