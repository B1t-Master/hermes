FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:stable-alpine
# Sane defaults; overridden at runtime (e.g. BACKEND_HOST=backend.railway.internal
# on Railway). Also supplied to the envsubst step that renders the nginx template.
ENV BACKEND_HOST=backend \
    BACKEND_PORT=8000
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
