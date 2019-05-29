echo "Removing container named fp"
docker stop fp
docker rm fp
echo "Running the image"
docker run --runtime=nvidia -d -p 8888:8888 --name fp \
--mount source=data,target=/data \
--mount type=bind,source=$HOME/fp_img,target=/fp_img,readonly \
--mount type=bind,source=$HOME/notebooks,target=/tf/notebooks \
--restart unless-stopped \
fp_analysis


echo "Waiting the server to run for 5 seconds..."
sleep 5

# docker logs fp 2>&1 | grep "token="
docker exec fp jupyter notebook list | grep "token="
