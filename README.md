

## Build standalone data-lake binary
```bash
docker build -t py38builder .
docker create --name tmpbuild py38builder
docker cp tmpbuild:/app/dist/main ./main
docker rm tmpbuild
```

