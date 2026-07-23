IMAGE := ghcr.io/zfdupont/pokerbot-trainer
PLATFORM := linux/amd64

.PHONY: docker-build docker-push install-hooks

docker-build:
	docker buildx build --platform $(PLATFORM) \
		-f docker/Dockerfile \
		-t $(IMAGE):latest \
		.

docker-push: docker-build
	docker push $(IMAGE):latest

install-hooks:
	@cp scripts/hooks/pre-push .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "pre-push hook installed"
