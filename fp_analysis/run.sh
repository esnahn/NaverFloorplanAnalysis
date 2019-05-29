echo "Removing container named fp"
docker stop fp
docker rm fp
docker run --runtime=nvidia -d -p 8888:8888 --name fp \
--mount source=data,target=/data \
--mount type=bind,source=$HOME/fp_img,target=/fp_img,readonly \
--mount type=bind,source=$HOME/notebooks,target=/tf/notebooks \
fp_analysis
docker exec fp jupyter notebook list
docker logs fp 2>&1 | grep "token="
