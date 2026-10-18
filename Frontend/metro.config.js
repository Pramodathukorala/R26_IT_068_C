const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

const defaultConfig = getDefaultConfig(__dirname);

/**
 * Metro's file watcher walks into each native module's Gradle output folder.
 * Those directories are created and deleted while an Android build is running,
 * so watching them crashes Metro with `ENOENT: watch ...\android\build\...`
 * whenever the bundler and Gradle run at the same time. Nothing under a
 * package's `android/build` is ever part of the JS bundle, so exclude it.
 *
 * Paths reaching blockList use forward slashes on every platform.
 */
const gradleOutputs = /.*[\\/]android[\\/](build|\.cxx)[\\/].*/;

const existing = defaultConfig.resolver.blockList;
const blockList = existing
  ? [].concat(existing, gradleOutputs)
  : gradleOutputs;

module.exports = mergeConfig(defaultConfig, {
  resolver: { blockList },
});
