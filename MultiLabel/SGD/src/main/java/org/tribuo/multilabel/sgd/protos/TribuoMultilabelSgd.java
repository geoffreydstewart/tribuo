// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-multilabel-sgd.proto

package org.tribuo.multilabel.sgd.protos;

public final class TribuoMultilabelSgd {
  private TribuoMultilabelSgd() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_multilabel_sgd_FMMultiLabelModelProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_multilabel_sgd_FMMultiLabelModelProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_multilabel_sgd_MultiLabelLinearSGDProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_multilabel_sgd_MultiLabelLinearSGDProto_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\033tribuo-multilabel-sgd.proto\022\025tribuo.mu" +
      "ltilabel.sgd\032\021tribuo-core.proto\032\021tribuo-" +
      "math.proto\"\272\001\n\026FMMultiLabelModelProto\022-\n" +
      "\010metadata\030\001 \001(\0132\033.tribuo.core.ModelDataP" +
      "roto\022,\n\006params\030\002 \001(\0132\034.tribuo.math.Param" +
      "etersProto\0220\n\nnormalizer\030\003 \001(\0132\034.tribuo." +
      "math.NormalizerProto\022\021\n\tthreshold\030\004 \001(\001\"" +
      "\274\001\n\030MultiLabelLinearSGDProto\022-\n\010metadata" +
      "\030\001 \001(\0132\033.tribuo.core.ModelDataProto\022,\n\006p" +
      "arams\030\002 \001(\0132\034.tribuo.math.ParametersProt" +
      "o\0220\n\nnormalizer\030\003 \001(\0132\034.tribuo.math.Norm" +
      "alizerProto\022\021\n\tthreshold\030\004 \001(\001B$\n org.tr" +
      "ibuo.multilabel.sgd.protosP\001b\006proto3"
    };
    descriptor = com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          org.tribuo.protos.core.TribuoCore.getDescriptor(),
          org.tribuo.math.protos.TribuoMath.getDescriptor(),
        });
    internal_static_tribuo_multilabel_sgd_FMMultiLabelModelProto_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tribuo_multilabel_sgd_FMMultiLabelModelProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_multilabel_sgd_FMMultiLabelModelProto_descriptor,
        new java.lang.String[] { "Metadata", "Params", "Normalizer", "Threshold", });
    internal_static_tribuo_multilabel_sgd_MultiLabelLinearSGDProto_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_tribuo_multilabel_sgd_MultiLabelLinearSGDProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_multilabel_sgd_MultiLabelLinearSGDProto_descriptor,
        new java.lang.String[] { "Metadata", "Params", "Normalizer", "Threshold", });
    org.tribuo.protos.core.TribuoCore.getDescriptor();
    org.tribuo.math.protos.TribuoMath.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}